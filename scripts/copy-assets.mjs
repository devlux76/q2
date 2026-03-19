#!/usr/bin/env bun
/**
 * copy-assets.mjs — Copies index.html (with corrected script paths) and
 * style.css into the dist/ directory so it can be deployed as a self-contained
 * GitHub Pages site.
 *
 * Run automatically by `postbuild` after `bun build`.
 */

import { readFileSync, writeFileSync, cpSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

// Copy stylesheet
cpSync(join(root, 'style.css'), join(root, 'dist', 'style.css'));

// Copy HTML with script src rewritten from "dist/app.js" → "app.js"
// so that the deployed site resolves correctly when dist/ is the web root.
const html = readFileSync(join(root, 'index.html'), 'utf8')
  .replace(/src\s*=\s*["']\s*dist\/app\.js\s*["']/, 'src="app.js"');
writeFileSync(join(root, 'dist', 'index.html'), html);
