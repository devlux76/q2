import { describe, expect, it, beforeEach, vi } from 'vitest';
import {
  storeFile,
  storeFromUrl,
  listStoredFiles,
  getStoredFile,
  deleteStoredFile,
  isOpfsAvailable,
} from '../src/opfs';

describe('opfs storage layer', () => {
  beforeEach(() => {
    localStorage.clear();
    // Ensure test environment has no OPFS API by default.
    delete (navigator as any).storage;
    delete (window as any).originPrivateFileSystem;
  });

  it('reports OPFS availability correctly when APIs are missing', () => {
    expect(isOpfsAvailable()).toBe(false);
  });

  it('stores a blob and records metadata mapping', async () => {
    const blob = new Blob(['hello world'], { type: 'text/plain' });
    const meta = await storeFile(blob, 'hello.txt');

    expect(meta.name).toBe('hello.txt');
    expect(meta.size).toBe(blob.size);
    expect(meta.hash).toMatch(/^[0-9a-f]{64}$/);

    const list = listStoredFiles();
    expect(list.length).toBe(1);
    expect(list[0].hash).toBe(meta.hash);
    expect(list[0].name).toBe('hello.txt');
  });

  it('can store a file from a URL and uses URL filename when name missing', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new TextEncoder().encode('zip data').buffer,
    });

    const meta = await storeFromUrl('https://example.com/archive.zip');
    expect(meta.name).toBe('archive.zip');
    expect(meta.url).toBe('https://example.com/archive.zip');

    const list = listStoredFiles();
    expect(list[0].hash).toBe(meta.hash);
  });

  it('getStoredFile returns null when OPFS is not available', async () => {
    const blob = new Blob(['abc']);
    const meta = await storeFile(blob, 'abc.txt');
    const data = await getStoredFile(meta.hash);
    expect(data).toBeNull();
  });

  it('deleteStoredFile removes mapping', async () => {
    const meta = await storeFile(new Blob(['x']), 'x.txt');
    expect(listStoredFiles().length).toBe(1);
    await deleteStoredFile(meta.hash);
    expect(listStoredFiles().length).toBe(0);
  });
});
