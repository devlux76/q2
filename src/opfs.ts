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
  const buffer = data instanceof Uint8Array ? data : new Uint8Array(data);
  const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(hashBuf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

async function getOpfsRoot(): Promise<FileSystemDirectoryHandle | null> {
  // Modern browsers: navigator.storage.getDirectory()
  // Chrome/Edge: self.originPrivateFileSystem
  const nav = (navigator as any) as { storage?: unknown };
  if (nav.storage && typeof (nav.storage as any).getDirectory === 'function') {
    // @ts-expect-error: TS lacks this method on Storage interface
    return await (nav.storage as any).getDirectory();
  }
  // Some environments expose originPrivateFileSystem directly.
  const win = window as any;
  if (win.originPrivateFileSystem) {
    return win.originPrivateFileSystem as FileSystemDirectoryHandle;
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
  await writable.write(data);
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

async function deleteOpfsFile(path: string): Promise<void> {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir) throw new Error('OPFS is not available in this environment');
  const name = path.replace(/^\/+|\/+$/g, '');
  // @ts-expect-error: removeEntry is not yet in TypeScript lib.
  if (typeof (dir as any).removeEntry === 'function') {
    // spec: removeEntry(name, { recursive: false })
    await (dir as any).removeEntry(name, { recursive: false });
  } else if (typeof (dir as any).remove === 'function') {
    await (dir as any).remove(name);
  } else {
    // No deletion support; ignore.
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
  const buffer = await (file instanceof File ? file.arrayBuffer() : file.arrayBuffer());
  const hash = await digestHex(buffer);
  const primaryName = ensureName(name, (file as File).name ?? hash);
  const mapping = loadMapping();
  const now = Date.now();
  const meta: StoredFileMeta = {
    hash,
    name: primaryName,
    size: buffer.byteLength,
    created: now,
    url,
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
  const nav = (navigator as any) as { storage?: unknown };
  if (nav.storage && typeof (nav.storage as any).getDirectory === 'function') return true;
  const win = window as any;
  return Boolean(win.originPrivateFileSystem);
}
