/**
 * Minimal static file server for Playwright E2E tests.
 *
 * Serves the project root (index.html, style.css, dist/, etc.) on port 4173.
 * This is intentionally simple — no framework, no bundler middleware.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';

const PORT = 4173;
const ROOT = resolve(process.cwd());

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png':  'image/png',
  '.svg':  'image/svg+xml',
  '.gif':  'image/gif',
  '.map':  'application/octet-stream',
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', `http://localhost:${PORT}`);
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === '/') pathname = '/index.html';

    // Resolve against ROOT and ensure the result stays inside the project.
    const filePath = resolve(ROOT, '.' + pathname);
    if (!filePath.startsWith(ROOT + sep) && filePath !== ROOT) {
      res.statusCode = 400;
      res.end('Bad request');
      return;
    }

    const data = await readFile(filePath);
    const ext = extname(filePath).toLowerCase();
    res.setHeader('Content-Type', MIME[ext] ?? 'application/octet-stream');
    /* Required for SharedArrayBuffer / WASM threads (COOP/COEP). */
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.end(data);
  } catch {
    res.statusCode = 404;
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`E2E static server listening on http://localhost:${PORT}`);
});
