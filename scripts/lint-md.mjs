#!/usr/bin/env bun
/**
 * lint-md.mjs — Lints Markdown files for encoding issues that break GitHub
 * rendering of KaTeX math and Mermaid diagrams.
 *
 * Checks performed:
 *   1. Emoji characters (U+1F000+) inside LaTeX $...$ or $$...$$ blocks —
 *      KaTeX cannot render emoji inside \text{} or math mode.
 *   2. Unicode MINUS SIGN (U+2212 −) inside LaTeX math blocks — use ASCII
 *      hyphen-minus (-) inside \text{...} and in Mermaid labels instead.
 *   3. Rare Unicode subscript/modifier letters (U+1D00–U+1D9F, U+2080–U+20A0)
 *      in Mermaid diagram source — these characters have limited renderer
 *      support and silently corrupt Mermaid output.
 *   4. Unicode MINUS SIGN (U+2212) anywhere in Mermaid blocks — diagram
 *      labels should use ASCII hyphen-minus.
 *
 * Usage:
 *   bun scripts/lint-md.mjs [file.md ...]      # lint specific files
 *   bun scripts/lint-md.mjs                    # lint *.md in repo root
 *
 * Exit code 1 if any errors are found (suitable for CI and pre-commit hooks).
 */

import { readFileSync, readdirSync } from 'fs';
import { resolve, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

// ── helpers ──────────────────────────────────────────────────────────────────

/** Split content on $$ boundaries; odd-indexed parts are display math. */
function displayMathBlocks(content) {
  const parts = content.split('$$');
  const blocks = [];
  for (let i = 1; i < parts.length; i += 2) {
    const start = parts.slice(0, i).join('$$').length + 2; // byte offset approx
    blocks.push({ text: parts[i], partIndex: i });
  }
  return blocks;
}

/** Extract the content and approximate line number of each ```mermaid block. */
function mermaidBlocks(lines) {
  const blocks = [];
  let inside = false;
  let startLine = 0;
  let buf = [];
  for (let i = 0; i < lines.length; i++) {
    if (!inside && lines[i].trim() === '```mermaid') {
      inside = true;
      startLine = i + 1; // 1-based
      buf = [];
    } else if (inside && lines[i].trim() === '```') {
      blocks.push({ lines: buf, startLine });
      inside = false;
    } else if (inside) {
      buf.push({ text: lines[i], lineNo: i + 1 });
    }
  }
  return blocks;
}

/**
 * Compute 1-based line number of a character offset within content.
 * Used to report line numbers for math-block violations.
 */
function lineOf(content, offset) {
  return content.slice(0, offset).split('\n').length;
}

// ── rules ─────────────────────────────────────────────────────────────────────

/**
 * Rule 1 & 2: scan display math ($$...$$) for disallowed Unicode.
 * Returns array of { line, col, message } violations.
 */
function checkDisplayMath(content, filePath) {
  const violations = [];
  const parts = content.split('$$');

  let offset = 0;
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      // inside a display math block
      const block = parts[i];
      for (let j = 0; j < block.length; j++) {
        const cp = block.codePointAt(j);
        if (cp > 0xffff) j++; // surrogate pair step-over

        if (cp >= 0x1f000) {
          const lineNo = lineOf(content, offset + 2 + j);
          violations.push({
            file: filePath, line: lineNo,
            message: `Emoji U+${cp.toString(16).toUpperCase()} ('${String.fromCodePoint(cp)}') inside LaTeX display math — KaTeX cannot render emoji; use \\text{word} instead`,
          });
        } else if (cp === 0x2212) {
          // U+2212 MINUS SIGN inside math — check if it's inside \text{...}
          const before = block.slice(0, j);
          const inText = /\\text\{[^}]*$/.test(before);
          if (inText) {
            const lineNo = lineOf(content, offset + 2 + j);
            violations.push({
              file: filePath, line: lineNo,
              message: `Unicode MINUS SIGN U+2212 ('−') inside \\text{} in display math — use ASCII hyphen-minus (-) instead`,
            });
          }
        }
      }
    }
    offset += parts[i].length + 2; // +2 for the $$ delimiter
  }
  return violations;
}

/**
 * Rule 2 & 3: scan Mermaid blocks for disallowed Unicode.
 */
function checkMermaid(content, filePath) {
  const lines = content.split('\n');
  const violations = [];

  for (const block of mermaidBlocks(lines)) {
    for (const { text, lineNo } of block.lines) {
      for (let j = 0; j < text.length; ) {
        const cp = text.codePointAt(j);
        const advance = cp > 0xffff ? 2 : 1;

        // U+2212 MINUS SIGN — should be ASCII hyphen-minus in Mermaid labels
        if (cp === 0x2212) {
          violations.push({
            file: filePath, line: lineNo,
            message: `Unicode MINUS SIGN U+2212 ('−') in Mermaid diagram — use ASCII hyphen-minus (-) instead`,
          });
        }

        // Rare subscript/modifier letters: U+1D00–U+1D9F and U+2080–U+209F
        // These include ᵢ (U+1D62), ᵣ (U+1D63), ᵥ (U+1D65), ₑ (U+2091), etc.
        if (
          (cp >= 0x1d00 && cp <= 0x1d9f) ||
          (cp >= 0x2080 && cp <= 0x209f)
        ) {
          violations.push({
            file: filePath, line: lineNo,
            message: `Rare Unicode modifier/subscript letter U+${cp.toString(16).toUpperCase()} ('${String.fromCodePoint(cp)}') in Mermaid — use plain ASCII text instead`,
          });
        }

        // Emoji in Mermaid (also unusual)
        if (cp >= 0x1f000) {
          violations.push({
            file: filePath, line: lineNo,
            message: `Emoji U+${cp.toString(16).toUpperCase()} ('${String.fromCodePoint(cp)}') in Mermaid diagram — use plain ASCII text instead`,
          });
        }

        j += advance;
      }
    }
  }
  return violations;
}

// ── main ──────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const files =
  args.length > 0
    ? args.map(f => resolve(f))
    : readdirSync(root)
        .filter(f => extname(f) === '.md')
        .map(f => join(root, f));

let totalErrors = 0;

for (const filePath of files) {
  let content;
  try {
    content = readFileSync(filePath, 'utf8');
  } catch {
    console.error(`lint-md: cannot read ${filePath}`);
    process.exit(1);
  }

  const violations = [
    ...checkDisplayMath(content, filePath),
    ...checkMermaid(content, filePath),
  ];

  for (const { file, line, message } of violations) {
    const rel = file.replace(root + '/', '');
    console.error(`${rel}:${line}: error: ${message}`);
    totalErrors++;
  }
}

if (totalErrors > 0) {
  console.error(`\nlint-md: ${totalErrors} error(s) found.`);
  process.exit(1);
} else {
  console.log(`lint-md: ${files.length} file(s) checked, no errors.`);
}
