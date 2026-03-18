#!/usr/bin/env node
import { createServer } from 'http';
import { readFile, mkdir } from 'fs/promises';
import { join, extname } from 'path';
import { chromium } from 'playwright';

const root = process.cwd();
const outputDir = join(root, 'docs', 'screenshots');

function contentType(filePath) {
  const ext = extname(filePath).toLowerCase();
  switch (ext) {
    case '.html': return 'text/html; charset=utf-8';
    case '.js': return 'application/javascript; charset=utf-8';
    case '.css': return 'text/css; charset=utf-8';
    case '.json': return 'application/json; charset=utf-8';
    case '.png': return 'image/png';
    case '.svg': return 'image/svg+xml';
    case '.map': return 'application/octet-stream';
    default: return 'application/octet-stream';
  }
}

async function serve() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      let pathname = url.pathname;
      if (pathname === '/') pathname = '/index.html';

      // Prevent directory traversal.
      if (pathname.includes('..')) {
        res.statusCode = 400;
        res.end('Bad request');
        return;
      }

      const filePath = join(root, pathname);
      const data = await readFile(filePath);
      res.setHeader('Content-Type', contentType(filePath));
      res.end(data);
    } catch (err) {
      res.statusCode = 404;
      res.end('Not found');
    }
  });

  await new Promise((resolve) => server.listen(0, resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Failed to start server');
  const url = `http://localhost:${address.port}`;
  return { server, url };
}

async function main() {
  await mkdir(outputDir, { recursive: true });

  const { server, url } = await serve();
  const browser = await chromium.launch();
  const page = await browser.newPage();

  const workerScript = `
self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'load') {
    postMessage({ type: 'status', status: 'loading', detail: 'Fetching model weights…' });
    const total = 1200;
    [120, 420, 820, 1200].forEach((loaded, i) => {
      setTimeout(() => {
        postMessage({ type: 'progress', file: 'model.onnx', loaded, total });
      }, 150 + i * 150);
    });
    setTimeout(() => postMessage({ type: 'status', status: 'ready' }), 900);
  }
  if (msg.type === 'generate') {
    postMessage({ type: 'status', status: 'generating' });
    postMessage({ type: 'token', token: 'H' });
    postMessage({ type: 'token', token: 'i' });
    const seqLen = 1;
    const hiddenDim = 16;
    const data = new Float32Array(seqLen * hiddenDim);
    for (let i = 0; i < data.length; i++) data[i] = Math.sin(i / 4);
    postMessage({ type: 'embedding', data: data.buffer, seqLen, hiddenDim, dtype: 'fp32' }, [data.buffer]);
    postMessage({ type: 'done' });
  }
};
`;

  await page.addInitScript({
    content: `window.__Q2_WORKER_URL__ = URL.createObjectURL(new Blob([${JSON.stringify(workerScript)}], { type: 'application/javascript' }));`,
  });

  await page.goto(url, { waitUntil: 'networkidle' });

  // Capture the warm load screen after a small delay to show download progress.
  await page.fill('#model-custom-id', 'test/model');
  await page.click('#load-btn');
  await page.waitForTimeout(500);

  await page.screenshot({ path: join(outputDir, 'load-screen.png'), fullPage: true });

  // Wait for the chat UI to appear.
  await page.waitForSelector('#chat-app:not(.hidden)');
  await page.screenshot({ path: join(outputDir, 'chat-empty.png'), fullPage: true });

  // Send a message to trigger an embedding and render the embedding panel.
  await page.fill('#user-input', 'Hello');
  await page.keyboard.press('Enter');
  await page.waitForSelector('#embedding-panel:not(.hidden)');
  await page.waitForTimeout(200); // allow Q² rendering to settle
  await page.screenshot({ path: join(outputDir, 'chat-embedding.png'), fullPage: true });

  await browser.close();
  server.close();
  console.log('Screenshots generated in', outputDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
