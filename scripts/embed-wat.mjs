#!/usr/bin/env node
/**
 * embed-wat.mjs — Updates the WASM_B64 constant in src/q2.ts with the
 * freshly-compiled bytes from src/q2.wasm.
 *
 * Run automatically by `npm run build:wat` after `wat2wasm`.
 */

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

// Read compiled WASM bytes
const wasmBytes = readFileSync(join(root, 'src', 'q2.wasm'));
const b64 = wasmBytes.toString('base64');

// Split into 80-char chunks for a readable TS string literal
const chunks = b64.match(/.{1,80}/g) ?? [];
const lines = chunks.map((chunk, i) => {
  const sep = i < chunks.length - 1 ? ' +' : ';';
  return `  '${chunk}'${sep}`;
});
const constBlock = `const WASM_B64 =\n${lines.join('\n')}`;

// Replace the existing WASM_B64 constant in q2.ts using line-based markers
// rather than a regex with nested repetition (avoids ReDoS).
const tsPath = join(root, 'src', 'q2.ts');
const src = readFileSync(tsPath, 'utf8');

const START_MARKER = 'const WASM_B64 =';
const start = src.indexOf(START_MARKER);
if (start === -1) {
  console.error('embed-wat.mjs: WASM_B64 constant not found in src/q2.ts');
  process.exit(1);
}
// Find the end of the constant: the first "'; at the start of a suffix after the declaration.
const afterDecl = src.indexOf('\n', start);
let end = src.indexOf("';", afterDecl);
if (end === -1) {
  console.error("embed-wat.mjs: closing '; not found after WASM_B64 in src/q2.ts");
  process.exit(1);
}
end += 2; // include the '; characters

const updated = src.slice(0, start) + constBlock + src.slice(end);

if (updated === src) {
  console.error('embed-wat.mjs: WASM_B64 block unchanged; check src/q2.ts format');
  process.exit(1);
}

writeFileSync(tsPath, updated);
console.log(`embed-wat.mjs: updated WASM_B64 (${wasmBytes.length} bytes → ${b64.length} chars base64)`);
