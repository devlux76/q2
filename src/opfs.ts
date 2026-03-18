/**
 * opfs.ts — Origin Private File System helper
 *
 * Provides a simple, browser-native key/value file store backed by the OPFS
 * when available. Files are stored under a per-origin directory and keyed by the
 * SHA-256 digest of their contents. A lightweight mapping stored in
 * localStorage preserves the original filename and metadata.
 *
 * Designed for: dropping a ZIP/model/corpus and later retrieving it by hash.
 */

export interface StoredFileMeta {
  /** SHA-256 hash of the file contents, hex-encoded. */
  hash: string;
  /** Original filename provided by the user or inferred from URL. */
  name: string;
  /** Number of bytes in the stored file. */
  size: number;
  /** Unix epoch ms when the file was stored. */
  created: number;
  /** Optional source URL (when imported via URL). */
  url?: string;
}

const OPFS_DIR = 'q2';
const LOCALSTORAGE_KEY = 'q2_opfs_file_map_v1';

function loadMapping(): Record<string, StoredFileMeta> {
  try {
    const raw = localStorage.getItem(LOCALSTORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, StoredFileMeta>) : {};
  } catch {
    return {};
  }
}

function saveMapping(mapping: Record<string, StoredFileMeta>): void {
  try {
    localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(mapping));
  } catch {
    // Ignore storage failures.
  }
}

async function digestHex(data: ArrayBuffer | Uint8Array): Promise<string> {
  // Normalize to a Uint8Array view. data instanceof Uint8Array uses a typed
  // check that works even in cross-realm (jsdom) contexts.
  const view = data instanceof Uint8Array ? data : new Uint8Array(data as ArrayBuffer);
  // Pass the view directly to Web Crypto to avoid an extra full-buffer copy.
  const hashBuf = await crypto.subtle.digest('SHA-256', view as BufferSource);
  return Array.from(new Uint8Array(hashBuf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

async function getOpfsRoot(): Promise<FileSystemDirectoryHandle | null> {
  // Modern browsers: navigator.storage.getDirectory()
  // Chrome/Edge: self.originPrivateFileSystem
  interface StorageWithOPFS { getDirectory(): Promise<FileSystemDirectoryHandle> }
  interface NavigatorWithOPFS { storage?: StorageWithOPFS }
  const nav = navigator as NavigatorWithOPFS;
  if (nav.storage && typeof nav.storage.getDirectory === 'function') {
    return await nav.storage.getDirectory();
  }
  // Some environments expose originPrivateFileSystem directly.
  interface WindowWithOPFS { originPrivateFileSystem?: FileSystemDirectoryHandle }
  const win = window as WindowWithOPFS;
  if (win.originPrivateFileSystem) {
    return win.originPrivateFileSystem;
  }
  return null;
}

async function ensureDir(pathSegments: string[]): Promise<FileSystemDirectoryHandle | null> {
  const root = await getOpfsRoot();
  if (!root) return null;
  let dir: FileSystemDirectoryHandle = root;
  for (const seg of pathSegments) {
    dir = await dir.getDirectoryHandle(seg, { create: true });
  }
  return dir;
}

async function writeOpfsFile(path: string, data: Uint8Array | ArrayBuffer): Promise<void> {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir) throw new Error('OPFS is not available in this environment');
  const name = path.replace(/^\/+|\/+$/g, '');
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  // Normalise to Uint8Array<ArrayBuffer>: FileSystemWritableFileStream.write()
  // requires BufferSource which excludes SharedArrayBuffer-backed typed arrays.
  const buf: Uint8Array<ArrayBuffer> = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data); // copy via ArrayLike<number> constructor → ArrayBuffer-backed
  await writable.write(buf);
  await writable.close();
}

async function readOpfsFile(path: string): Promise<Uint8Array> {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir) throw new Error('OPFS is not available in this environment');
  const name = path.replace(/^\/+|\/+$/g, '');
  const handle = await dir.getFileHandle(name, { create: false });
  const file = await handle.getFile();
  return new Uint8Array(await file.arrayBuffer());
}

interface FileSystemDirectoryHandleWithRemoveEntry extends FileSystemDirectoryHandle {
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
}

function hasRemoveEntry(
  dir: FileSystemDirectoryHandle,
): dir is FileSystemDirectoryHandleWithRemoveEntry {
  return typeof (dir as FileSystemDirectoryHandleWithRemoveEntry).removeEntry === 'function';
}

async function deleteOpfsFile(path: string): Promise<void> {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir) throw new Error('OPFS is not available in this environment');
  const name = path.replace(/^\/+|\/+$/g, '');
  if (hasRemoveEntry(dir)) {
    // spec: removeEntry(name, { recursive: false })
    await dir.removeEntry(name, { recursive: false });
  } else {
    // Legacy non-standard `.remove(name)` API present in some early Chrome builds.
    interface FileSystemDirectoryHandleWithRemove extends FileSystemDirectoryHandle {
      remove(name: string): Promise<void>;
    }
    const legacyDir = dir as FileSystemDirectoryHandleWithRemove;
    if (typeof legacyDir.remove === 'function') {
      await legacyDir.remove(name);
    }
    // No deletion support in this environment; ignore.
  }
}

function ensureName(name: string | undefined, fallback: string): string {
  if (name && name.trim()) return name.trim();
  return fallback;
}

export async function storeFile(
  file: File | Blob,
  name?: string,
  url?: string,
): Promise<StoredFileMeta> {
  const buffer = await file.arrayBuffer();
  const hash = await digestHex(buffer);
  const primaryName = ensureName(name, (file as File).name ?? hash);
  const mapping = loadMapping();
  const now = Date.now();
  const meta: StoredFileMeta = {
    hash,
    name: primaryName,
    size: buffer.byteLength,
    created: now,
    ...(url !== undefined && { url }),
  };
  mapping[hash] = meta;
  saveMapping(mapping);

  // Persist the bytes in OPFS if available.
  try {
    await writeOpfsFile(hash, buffer);
  } catch {
    // Fallback: no-op. Mapping still exists so the app can show it.
  }

  return meta;
}

export async function storeFromUrl(url: string, name?: string): Promise<StoredFileMeta> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const hash = await digestHex(buffer);

  const primaryName = ensureName(name, new URL(url).pathname.split('/').pop() || hash);
  const mapping = loadMapping();
  const now = Date.now();
  const meta: StoredFileMeta = { hash, name: primaryName, size: buffer.byteLength, created: now, url };
  mapping[hash] = meta;
  saveMapping(mapping);

  try {
    await writeOpfsFile(hash, buffer);
  } catch {
    // ignore
  }

  return meta;
}

export function listStoredFiles(): StoredFileMeta[] {
  return Object.values(loadMapping()).sort((a, b) => b.created - a.created);
}

export async function getStoredFile(hash: string): Promise<Uint8Array | null> {
  try {
    const data = await readOpfsFile(hash);
    return data;
  } catch {
    return null;
  }
}

export async function deleteStoredFile(hash: string): Promise<void> {
  const mapping = loadMapping();
  delete mapping[hash];
  saveMapping(mapping);
  try {
    await deleteOpfsFile(hash);
  } catch {
    // Ignore failures; mapping is authoritative.
  }
}

export function isOpfsAvailable(): boolean {
  interface StorageWithOPFS { getDirectory(): Promise<FileSystemDirectoryHandle> }
  interface NavigatorWithOPFS { storage?: StorageWithOPFS }
  interface WindowWithOPFS { originPrivateFileSystem?: FileSystemDirectoryHandle }
  const nav = navigator as NavigatorWithOPFS;
  if (nav.storage && typeof nav.storage.getDirectory === 'function') return true;
  const win = window as WindowWithOPFS;
  return Boolean(win.originPrivateFileSystem);
}
