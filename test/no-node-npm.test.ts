import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'fs';
import { join } from 'path';

const bannedPatterns = [
  /\bnpm\b/, // do not invoke npm
  /\bnpx\b/, // do not invoke npx
  /\bnode\b/, // do not invoke node (use bun instead)
];

/**
 * Files allowed to contain these patterns (e.g., in docs or dependency names).
 * Each entry is a regex tested against the file path.
 */
const allowList = [
  /package\.json$/, // dependency names may mention node
  /README\.md$/, // docs may mention node
];

function isAllowed(filePath: string): boolean {
  return allowList.some((re) => re.test(filePath));
}

function findScriptFiles(dir: string): string[] {
  const entries = readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const ent of entries) {
    const full = join(dir, ent.name);
    if (ent.isDirectory()) {
      files.push(...findScriptFiles(full));
    } else if (ent.isFile()) {
      files.push(full);
    }
  }
  return files;
}

describe('no node/npm/npx pipeline tooling', () => {
  it('does not invoke node/npm/npx in package scripts', () => {
    const pkg = JSON.parse(readFileSync(join(__dirname, '..', 'package.json'), 'utf8')) as {
      scripts?: Record<string, string>;
    };
    const scripts = pkg.scripts ?? {};
    for (const [, cmd] of Object.entries(scripts)) {
      for (const pat of bannedPatterns) {
        expect(cmd).not.toMatch(pat);
      }
    }
  });

  it('does not use node/npm/npx in scripts folder', () => {
    const scriptFiles = findScriptFiles(join(__dirname, '..', 'scripts'));
    for (const file of scriptFiles) {
      if (isAllowed(file)) continue;
      const content = readFileSync(file, 'utf8');
      for (const pat of bannedPatterns) {
        expect(content).not.toMatch(pat);
      }
    }
  });
});
