/**
 * chat-render.ts — Pure chat text rendering utilities
 *
 * Functions for parsing and formatting assistant response text:
 *  - Splits raw streamed text into <think> blocks and regular text
 *  - Strips think tags from history (avoid consuming context on re-use)
 *  - Formats plain text with minimal safe HTML (escaping, code blocks, bold, italic)
 */

export interface TextPart { type: 'text'; text: string }
export interface ThinkPart { type: 'think'; text: string }
export type Part = TextPart | ThinkPart;

/**
 * Split raw assistant text into alternating text/think segments.
 *
 * Handles both complete `<think>…</think>` blocks and an incomplete open
 * `<think>` tag that is still being streamed.
 */
export function splitThinkBlocks(raw: string): Part[] {
  const parts: Part[] = [];
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  let lastIndex = 0;
  let m: RegExpExecArray | null;

  while ((m = re.exec(raw)) !== null) {
    if (m.index > lastIndex) {
      parts.push({ type: 'text', text: raw.slice(lastIndex, m.index) });
    }
    parts.push({ type: 'think', text: m[1] ?? '' });
    lastIndex = re.lastIndex;
  }

  if (lastIndex < raw.length) {
    // Handle an open (incomplete) <think> tag while still streaming.
    const tail = raw.slice(lastIndex);
    const openTag = tail.indexOf('<think>');
    if (openTag !== -1) {
      if (openTag > 0) parts.push({ type: 'text', text: tail.slice(0, openTag) });
      parts.push({ type: 'think', text: tail.slice(openTag + 7) });
    } else {
      parts.push({ type: 'text', text: tail });
    }
  }

  return parts;
}

/**
 * Strip `<think>…</think>` blocks from raw text before adding to history.
 *
 * Removes complete blocks first, then strips any remaining open `<think>` tag
 * and everything following (handles generation cut-off mid-think).
 */
export function stripThinkTags(raw: string): string {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<think>[\s\S]*$/g, '')
    .trim();
}

/**
 * Minimal safe text formatter:
 *  - Escapes HTML entities
 *  - Renders ```code blocks``` and `inline code`
 *  - Converts **bold** and *italic*
 *  - Converts newlines to `<br>`
 */
export function escapeAndFormatText(text: string): string {
  // 1. Escape HTML special chars.
  let s = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // 2. Fenced code blocks ```…``` — temporarily replace with placeholders
  const codeBlocks: string[] = [];
  s = s.replace(/```([\s\S]*?)```/g, (_match, codeContent: string) => {
    const index = codeBlocks.length;
    codeBlocks.push(`<pre class="code-block"><code>${codeContent}</code></pre>`);
    return `__CODE_BLOCK_${index}__`;
  });

  // 3. Inline code `…`
  s = s.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

  // 4. Bold **…**
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // 5. Italic *…* (single asterisk, not inside **…**)
  s = s.replace(/(^|[^*])\*([^*\n]+)\*([^*]|$)/g, '$1<em>$2</em>$3');

  // 6. Newlines → <br> (only in non-code segments)
  s = s.replace(/\n/g, '<br>');

  // 7. Restore fenced code blocks
  s = s.replace(/__CODE_BLOCK_(\d+)__/g, (_match, index: string) => {
    const i = Number(index);
    return codeBlocks[i] ?? '';
  });

  return s;
}
