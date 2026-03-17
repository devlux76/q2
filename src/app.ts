/**
 * app.ts — Main-thread entry point
 *
 * Manages:
 *  • Worker lifecycle (load → idle → generating → idle)
 *  • Chat history (system prompt + user/assistant turns)
 *  • DOM updates (progressive token streaming, thinking collapse)
 *  • Embedding panel (previews Q² input vectors from the last LIV layer)
 */

import type {
  WorkerOutMsg,
  WorkerInMsg,
  ChatMessage,
  GenerationConfig,
  EmbeddingMsg,
} from './types.js';

// ─── Constants ─────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT =
  'You are a helpful, harmless, and honest AI assistant. ' +
  'Think carefully before answering.';

const DEFAULT_CONFIG: GenerationConfig = {
  max_new_tokens: 2048,
  temperature: 0.0,
  repetition_penalty: 1.1,
};

// ─── DOM references ────────────────────────────────────────────────────────────

function $<T extends HTMLElement>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (!el) throw new Error(`Element not found: ${selector}`);
  return el;
}

const loadScreen = $<HTMLDivElement>('#load-screen');
const loadStatus = $<HTMLParagraphElement>('#load-status');
const loadBar = $<HTMLDivElement>('#load-bar-fill');
const chatApp = $<HTMLDivElement>('#chat-app');
const messagesEl = $<HTMLDivElement>('#messages');
const inputEl = $<HTMLTextAreaElement>('#user-input');
const sendBtn = $<HTMLButtonElement>('#send-btn');
const stopBtn = $<HTMLButtonElement>('#stop-btn');
const embeddingPanel = $<HTMLDivElement>('#embedding-panel');
const embeddingCanvas = $<HTMLCanvasElement>('#embedding-canvas');
const embeddingStats = $<HTMLParagraphElement>('#embedding-stats');

// Settings controls
const maxTokensEl = $<HTMLInputElement>('#max-tokens');
const temperatureEl = $<HTMLInputElement>('#temperature');
const tempValueEl = $<HTMLSpanElement>('#temp-value');
const repPenaltyEl = $<HTMLInputElement>('#rep-penalty');
const repValueEl = $<HTMLSpanElement>('#rep-value');

// ─── Application state ─────────────────────────────────────────────────────────

let worker: Worker | null = null;
let modelReady = false;
let isGenerating = false;

/** Persistent conversation history sent to the model each turn. */
const history: ChatMessage[] = [
  { role: 'system', content: SYSTEM_PROMPT },
];

/** The DOM node for the currently-streaming assistant bubble. */
let activeBubble: HTMLDivElement | null = null;
/** Accumulated raw text for the current response (including <think> tags). */
let activeRawText = '';

// ─── Worker bootstrap ──────────────────────────────────────────────────────────

function initWorker(): void {
  worker = new Worker(new URL('./worker.js', import.meta.url), {
    type: 'module',
  });

  worker.addEventListener('message', (e: MessageEvent<WorkerOutMsg>) => {
    handleWorkerMessage(e.data);
  });

  worker.addEventListener('error', (e) => {
    showError(`Worker error: ${e.message}`);
  });

  postToWorker({ type: 'load' });
}

function postToWorker(msg: WorkerInMsg): void {
  worker?.postMessage(msg);
}

// ─── Worker message handler ────────────────────────────────────────────────────

function handleWorkerMessage(msg: WorkerOutMsg): void {
  switch (msg.type) {
    case 'status':
      onStatus(msg.status, msg.detail);
      break;
    case 'progress':
      onProgress(msg.file, msg.loaded, msg.total);
      break;
    case 'token':
      onToken(msg.token);
      break;
    case 'embedding':
      onEmbedding(msg);
      break;
    case 'done':
      onDone();
      break;
    case 'error':
      if (!modelReady) {
        // Surface errors on the load screen when the model is not yet ready,
        // so users don't get stuck on "Initializing…" without feedback.
        loadStatus.textContent = `Error loading model: ${msg.message}`;
        loadScreen.classList.remove('hidden');
        chatApp.classList.add('hidden');
      } else {
        showError(msg.message);
      }
      onDone();
      break;
  }
}

// ─── Worker event handlers ─────────────────────────────────────────────────────

function onStatus(
  status: 'loading' | 'ready' | 'generating' | 'idle',
  detail?: string,
): void {
  if (status === 'ready') {
    modelReady = true;
    loadScreen.classList.add('hidden');
    chatApp.classList.remove('hidden');
    inputEl.focus();
  } else if (status === 'loading') {
    loadStatus.textContent = detail ?? 'Loading model…';
  }
}

function onProgress(file: string, loaded: number, total: number): void {
  if (total > 0) {
    const pct = Math.round((loaded / total) * 100);
    loadBar.style.width = `${pct}%`;
    const statusText = file
      ? `Downloading ${file.split('/').pop() ?? file} — ${pct}%`
      : `Downloading… ${pct}%`;
    loadStatus.textContent = statusText;
    loadBar.setAttribute('aria-valuenow', String(pct));
    loadBar.setAttribute('aria-valuetext', statusText);
  } else {
    const statusText =
      file ? `Loading ${file.split('/').pop() ?? file}…` : 'Loading…';
    loadStatus.textContent = statusText;
    loadBar.removeAttribute('aria-valuenow');
    loadBar.setAttribute('aria-valuetext', statusText);
  }
}

// Throttle bubble re-renders during streaming to at most once per animation frame.
let tokenRenderScheduled = false;

function scheduleBubbleRender(): void {
  if (!activeBubble) {
    // Nothing to render; clear any pending schedule.
    tokenRenderScheduled = false;
    return;
  }

  if (tokenRenderScheduled) return;
  tokenRenderScheduled = true;

  requestAnimationFrame(() => {
    // activeBubble / activeRawText may have changed since scheduling.
    if (!activeBubble) {
      tokenRenderScheduled = false;
      return;
    }
    renderBubble(activeBubble, activeRawText);
    scrollToBottom();
    tokenRenderScheduled = false;
  });
}

function onToken(token: string): void {
  if (!activeBubble) return;
  activeRawText += token;
  scheduleBubbleRender();
}

function onEmbedding(msg: EmbeddingMsg): void {
  // Display the last-LIV-layer embedding as a heat-map preview.
  // The actual Q² WASM quantisation kernel will be wired here.
  embeddingPanel.classList.remove('hidden');
  renderEmbeddingHeatmap(msg.data, msg.seqLen, msg.hiddenDim);
  embeddingStats.textContent =
    `Shape: [${msg.seqLen} × ${msg.hiddenDim}]  ` +
    `min=${min(msg.data).toFixed(3)}  max=${max(msg.data).toFixed(3)}`;
}

function onDone(): void {
  isGenerating = false;
  sendBtn.disabled = false;
  sendBtn.classList.remove('hidden');
  stopBtn.classList.add('hidden');

  if (activeBubble) {
    // Strip <think>…</think> blocks before adding to history so that reasoning
    // traces don't consume context on subsequent turns.
    history.push({ role: 'assistant', content: stripThinkTags(activeRawText) });
    // Ensure the final render is canonical.
    renderBubble(activeBubble, activeRawText);
    activeBubble = null;
    activeRawText = '';
  }

  inputEl.disabled = false;
  inputEl.focus();
}

// ─── User interaction ──────────────────────────────────────────────────────────

function sendMessage(): void {
  const text = inputEl.value.trim();
  if (!text || !modelReady || isGenerating) return;

  // Append user message to history and render it.
  history.push({ role: 'user', content: text });
  appendUserBubble(text);
  inputEl.value = '';
  autoResizeTextarea();

  // Prepare a new assistant bubble for streaming.
  activeBubble = appendAssistantBubble();
  activeRawText = '';

  isGenerating = true;
  sendBtn.disabled = true;
  sendBtn.classList.add('hidden');
  stopBtn.classList.remove('hidden');
  inputEl.disabled = true;

  postToWorker({
    type: 'generate',
    messages: history.slice(), // send a snapshot
    config: readConfig(),
  });
}

function stopGeneration(): void {
  postToWorker({ type: 'abort' });
}

function readConfig(): GenerationConfig {
  return {
    max_new_tokens: parseInt(maxTokensEl.value, 10) || DEFAULT_CONFIG.max_new_tokens,
    temperature: parseFloat(temperatureEl.value) || DEFAULT_CONFIG.temperature,
    repetition_penalty: parseFloat(repPenaltyEl.value) || DEFAULT_CONFIG.repetition_penalty,
  };
}

// ─── DOM helpers ───────────────────────────────────────────────────────────────

function appendUserBubble(text: string): void {
  const row = document.createElement('div');
  row.className = 'message-row user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble user';
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function appendAssistantBubble(): HTMLDivElement {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant';
  // Show a blinking cursor while streaming.
  bubble.innerHTML = '<span class="cursor"></span>';
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return bubble;
}

/**
 * Render the raw assistant text into the bubble.
 *
 * <think>…</think> blocks are rendered as a collapsed details/summary so the
 * reasoning trace is accessible but not visually dominant.
 */
function renderBubble(bubble: HTMLDivElement, raw: string): void {
  // Split out <think>…</think> blocks (LFM2.5-Thinking model).
  const parts = splitThinkBlocks(raw);
  bubble.innerHTML = '';

  for (const part of parts) {
    if (part.type === 'think') {
      const details = document.createElement('details');
      details.className = 'think-block';
      const summary = document.createElement('summary');
      summary.textContent = '💭 Thinking…';
      const pre = document.createElement('pre');
      pre.className = 'think-content';
      pre.textContent = part.text;
      details.appendChild(summary);
      details.appendChild(pre);
      bubble.appendChild(details);
    } else {
      const span = document.createElement('span');
      span.innerHTML = escapeAndFormatText(part.text);
      bubble.appendChild(span);
    }
  }

  // Add streaming cursor if still generating.
  if (isGenerating) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);
  }
}

interface TextPart { type: 'text'; text: string }
interface ThinkPart { type: 'think'; text: string }
type Part = TextPart | ThinkPart;

function splitThinkBlocks(raw: string): Part[] {
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

function stripThinkTags(raw: string): string {
  // Remove complete <think>...</think> blocks first, then strip any remaining
  // open <think> tag and everything that follows (in case generation was cut off).
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
 *  - Converts newlines to <br>
 */
function escapeAndFormatText(text: string): string {
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

function scrollToBottom(): void {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function autoResizeTextarea(): void {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
}

function showError(message: string): void {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble error';
  bubble.textContent = `⚠️ ${message}`;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

// ─── Embedding heat-map ────────────────────────────────────────────────────────

/**
 * Renders a tiny heat-map of the last-LIV-layer embeddings.
 * One column per sequence position, one row per hidden dimension bin.
 * Colour: blue (negative) → white (zero) → red (positive).
 */
function renderEmbeddingHeatmap(
  data: Float32Array,
  seqLen: number,
  hiddenDim: number,
): void {
  const canvas = embeddingCanvas;
  const W = Math.min(seqLen, canvas.clientWidth || 320);
  const H = 64; // fixed display height

  canvas.width = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rowsPerCell = Math.ceil(hiddenDim / H);
  const colsPerCell = Math.ceil(seqLen / W);

  const minVal = min(data);
  const maxVal = max(data);
  const range = maxVal - minVal || 1;

  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      // Average values in the bin.
      let sum = 0;
      let count = 0;
      for (let d = row * rowsPerCell; d < Math.min((row + 1) * rowsPerCell, hiddenDim); d++) {
        for (let s = col * colsPerCell; s < Math.min((col + 1) * colsPerCell, seqLen); s++) {
          sum += data[s * hiddenDim + d] ?? 0;
          count++;
        }
      }
      const v = count ? sum / count : 0;
      const t = (v - minVal) / range; // 0..1

      // Blue → White → Red colour map.
      const r = t > 0.5 ? 255 : Math.round(t * 2 * 255);
      const g = t > 0.5 ? Math.round((1 - t) * 2 * 255) : Math.round(t * 2 * 255);
      const b = t < 0.5 ? 255 : Math.round((1 - t) * 2 * 255);

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(col, row, 1, 1);
    }
  }
}

function min(arr: Float32Array): number {
  let m = Infinity;
  for (const v of arr) if (v < m) m = v;
  return m;
}

function max(arr: Float32Array): number {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  return m;
}

// ─── Event listeners ───────────────────────────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);

inputEl.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', autoResizeTextarea);

temperatureEl.addEventListener('input', () => {
  tempValueEl.textContent = parseFloat(temperatureEl.value).toFixed(2);
});

repPenaltyEl.addEventListener('input', () => {
  repValueEl.textContent = parseFloat(repPenaltyEl.value).toFixed(2);
});

// ─── Start ─────────────────────────────────────────────────────────────────────

initWorker();
