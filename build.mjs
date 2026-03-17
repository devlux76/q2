// Build script: bundles src/app.ts and src/worker.ts into dist/
import * as esbuild from 'esbuild';
import { argv } from 'process';
import { mkdirSync } from 'fs';

const watch = argv.includes('--watch');

mkdirSync('dist', { recursive: true });

/** @type {import('esbuild').BuildOptions} */
const shared = {
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
  sourcemap: true,
  minify: !watch,
};

if (watch) {
  const appCtx = await esbuild.context({ ...shared, entryPoints: ['src/app.ts'], outfile: 'dist/app.js' });
  const workerCtx = await esbuild.context({ ...shared, entryPoints: ['src/worker.ts'], outfile: 'dist/worker.js' });
  await Promise.all([appCtx.watch(), workerCtx.watch()]);
  console.log('Watching for changes...');
} else {
  await Promise.all([
    esbuild.build({ ...shared, entryPoints: ['src/app.ts'], outfile: 'dist/app.js' }),
    esbuild.build({ ...shared, entryPoints: ['src/worker.ts'], outfile: 'dist/worker.js' }),
  ]);
  console.log('Build complete → dist/');
}
