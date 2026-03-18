/**
 * app.ts — Main-thread entry point
 *
 * Manages:
 *  • Model discovery (live HuggingFace Hub API search + custom HF URN)
 *  • App settings (API token, ONNX dtype, library filter) persisted to localStorage
 *  • Worker lifecycle (load → idle → generating → idle)
 *  • Chat history (system prompt + user/assistant turns)
 *  • DOM updates (progressive token streaming, thinking collapse)
 *  • Embedding panel (Q² kernel: mean-pool, L2-norm, quantise → packed bytes + key)
 */

// Allow tests to override the worker entrypoint (e.g. a blob URL) and to
// prevent auto-start while the test configures globals.
declare global {
  var __Q2_SKIP_AUTO_INIT__: boolean | undefined;
  var __Q2_WORKER_URL__: string | undefined;
}

import type {
  WorkerOutMsg,
  WorkerInMsg,
  ChatMessage,
  GenerationConfig,
  EmbeddingMsg,
} from './types.js';
import {
  getKernel,
  meanPoolAndNormalise,
  q2EncodeDirect,
  DTYPE_TO_Q2,
  Q2_DTYPE_FP32,
  Q2_INPUT_OFFSET,
  Q2_OUTPUT_OFFSET,
} from './q2.js';
import {
  deleteStoredFile,
  getStoredFile,
  isOpfsAvailable,
  listStoredFiles,
  storeFile,
  storeFromUrl,
} from './opfs.js';
import { AppSettings, loadSettings, saveSettings } from './settings.js';
export { loadSettings, saveSettings };

// ─── HuggingFace Hub API ───────────────────────────────────────────────────────

/** Model record returned by the HF Hub API. */
export interface HFModel {
  id: string;
  downloads: number;
  likes: number;
  tags: string[];
}

/** Format a large number as a compact string (e.g. 1234567 → "1.2M"). */
export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/**
 * Fetch text-generation models from the HuggingFace Hub API.
 * No API key is required for public models; providing one enables private
 * models and increases the rate limit.
 */
export async function fetchHFModels(query: string, settings: AppSettings): Promise<HFModel[]> {
  const params = new URLSearchParams({
    pipeline_tag: 'text-generation',
    sort: 'downloads',
    direction: '-1',
    limit: '20',
  });
  if (query.trim()) params.set('search', query.trim());
  if (settings.filterLibrary) params.set('library', settings.filterLibrary);

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (settings.apiToken) headers['Authorization'] = `Bearer ${settings.apiToken}`;

  const res = await fetch(`https://huggingface.co/api/models?${params}`, { headers });
  if (!res.ok) throw new Error(`HF API ${res.status}: ${res.statusText}`);
  return (await res.json()) as HFModel[];
}

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

// Local file store UI (OPFS)
const localFileDrop = $<HTMLDivElement>('#local-file-drop');
const localFileUrl = $<HTMLInputElement>('#local-file-url');
const localFileAddBtn = $<HTMLButtonElement>('#local-file-add');
const localFilesList = $<HTMLUListElement>('#local-files-list');

const localFileDropDefaultText = localFileDrop.textContent ?? '';

// Generation controls (sidebar)
const maxTokensEl = $<HTMLInputElement>('#max-tokens');
const temperatureEl = $<HTMLInputElement>('#temperature');
const tempValueEl = $<HTMLSpanElement>('#temp-value');
const repPenaltyEl = $<HTMLInputElement>('#rep-penalty');
const repValueEl = $<HTMLSpanElement>('#rep-value');

// Model picker phase elements
const modelPickerEl = $<HTMLDivElement>('#model-picker');
const loadProgressEl = $<HTMLDivElement>('#load-progress');
const modelSearchEl = $<HTMLInputElement>('#model-search');
const modelListEl = $<HTMLUListElement>('#model-list');
const modelCustomIdEl = $<HTMLInputElement>('#model-custom-id');
const loadBtnEl = $<HTMLButtonElement>('#load-btn');
const headerTitleEl = $<HTMLSpanElement>('#header-title');
const sidebarModelTagEl = $<HTMLSpanElement>('#sidebar-model-tag');

// ─── Application state ─────────────────────────────────────────────────────────

export let worker: Worker | null = null;
let modelReady = false;
let isGenerating = false;

/** Loaded and applied before the first HF API call or model load. */
let currentSettings: AppSettings = loadSettings();

/**
 * The model ID selected in the picker or entered via the custom field.
 * Empty string means "nothing selected yet" (load button is disabled in that state).
 */
export let selectedModelId = '';

/** Persistent conversation history sent to the model each turn. */
const history: ChatMessage[] = [
  { role: 'system', content: SYSTEM_PROMPT },
];

/** The DOM node for the currently-streaming assistant bubble. */
let activeBubble: HTMLDivElement | null = null;
/** Accumulated raw text for the current response (including <think> tags). */
let activeRawText = '';

// ─── Model picker ──────────────────────────────────────────────────────────────

let searchTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Fetch models from HF Hub and render them into the list.
 * On the initial (empty-query) load, auto-selects the first result so the
 * Load button is immediately enabled.
 */
async function refreshModelList(query: string, autoSelectFirst = false): Promise<void> {
  modelListEl.innerHTML = '<li class="model-list-status">Searching models…</li>';

  try {
    const models = await fetchHFModels(query, currentSettings);

    if (models.length === 0) {
      modelListEl.innerHTML =
        '<li class="model-list-status">No models found. Try a different search.</li>';
      return;
    }

    modelListEl.innerHTML = '';
    for (const model of models) {
      renderModelItem(model);
    }

    // Auto-select the top result on the initial load so the Load button
    // is immediately usable without requiring an explicit click.
    if (autoSelectFirst && !selectedModelId && models[0]) {
      selectModel(models[0].id);
    }
  } catch (err) {
    const li = document.createElement('li');
    li.className = 'model-list-status model-list-error';
    li.textContent = String(err);
    const retryBtn = document.createElement('button');
    retryBtn.className = 'model-list-retry';
    retryBtn.textContent = 'Retry';
    retryBtn.addEventListener('click', () => void refreshModelList(query, autoSelectFirst));
    li.appendChild(retryBtn);
    modelListEl.innerHTML = '';
    modelListEl.appendChild(li);
  }
}

function renderModelItem(model: HFModel): void {
  const li = document.createElement('li');
  li.role = 'option';
  li.className = 'model-item' + (model.id === selectedModelId ? ' selected' : '');
  li.setAttribute('aria-selected', model.id === selectedModelId ? 'true' : 'false');
  li.dataset['modelId'] = model.id;

  const slashIdx = model.id.indexOf('/');
  const author = slashIdx !== -1 ? model.id.slice(0, slashIdx) : '';
  const name = slashIdx !== -1 ? model.id.slice(slashIdx + 1) : model.id;

  const info = document.createElement('div');
  info.className = 'model-item-info';

  const labelEl = document.createElement('span');
  labelEl.className = 'model-item-label';
  labelEl.textContent = name;

  const authorEl = document.createElement('span');
  authorEl.className = 'model-item-author';
  authorEl.textContent = author;

  info.appendChild(labelEl);
  info.appendChild(authorEl);

  const stats = document.createElement('div');
  stats.className = 'model-item-stats';

  const dlEl = document.createElement('span');
  dlEl.className = 'model-item-stat';
  dlEl.title = 'Downloads';
  dlEl.textContent = `↓ ${formatCount(model.downloads)}`;

  const likeEl = document.createElement('span');
  likeEl.className = 'model-item-stat';
  likeEl.title = 'Likes';
  likeEl.textContent = `♥ ${formatCount(model.likes)}`;

  stats.appendChild(dlEl);
  stats.appendChild(likeEl);
  li.appendChild(info);
  li.appendChild(stats);
  li.addEventListener('click', () => selectModel(model.id));
  modelListEl.appendChild(li);
}

export function selectModel(modelId: string): void {
  selectedModelId = modelId;
  modelCustomIdEl.value = '';
  document.querySelectorAll<HTMLLIElement>('.model-item').forEach((item) => {
    const selected = item.dataset['modelId'] === modelId;
    item.classList.toggle('selected', selected);
    item.setAttribute('aria-selected', selected ? 'true' : 'false');
  });
  loadBtnEl.disabled = false;
}

export function initModelPicker(): void {
  // Kick off the initial model list fetch from HF Hub.
  void refreshModelList('', true);

  // Debounced live search as the user types.
  modelSearchEl.addEventListener('input', () => {
    const q = modelSearchEl.value;
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      void refreshModelList(q);
      searchTimer = null;
    }, 400);
  });

  // Typing a custom ID clears the list selection.
  modelCustomIdEl.addEventListener('input', () => {
    const val = modelCustomIdEl.value.trim();
    if (val) {
      document.querySelectorAll<HTMLLIElement>('.model-item').forEach((item) => {
        item.classList.remove('selected');
        item.setAttribute('aria-selected', 'false');
      });
      loadBtnEl.disabled = false;
    } else {
      loadBtnEl.disabled = !selectedModelId;
    }
  });

  modelCustomIdEl.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      triggerLoad();
    }
  });

  loadBtnEl.addEventListener('click', triggerLoad);
  initSettingsPanel();
  initLocalFileStore();
}

/** Wire up the collapsible settings panel and persist changes to localStorage. */
function initSettingsPanel(): void {
  const toggleBtn = $<HTMLButtonElement>('#settings-toggle');
  const panel = $<HTMLDivElement>('#settings-panel');
  const tokenEl = $<HTMLInputElement>('#hf-token');
  const dtypeEl = $<HTMLSelectElement>('#model-dtype');
  const libraryEl = $<HTMLSelectElement>('#filter-library');

  // Restore persisted values into the form.
  tokenEl.value = currentSettings.apiToken;
  dtypeEl.value = currentSettings.dtype;
  libraryEl.value = currentSettings.filterLibrary;

  toggleBtn.addEventListener('click', () => {
    const isHidden = panel.classList.toggle('hidden');
    toggleBtn.setAttribute('aria-expanded', String(!isHidden));
  });

  tokenEl.addEventListener('change', () => {
    currentSettings.apiToken = tokenEl.value.trim();
    saveSettings(currentSettings);
  });

  dtypeEl.addEventListener('change', () => {
    currentSettings.dtype = dtypeEl.value;
    saveSettings(currentSettings);
  });

  libraryEl.addEventListener('change', () => {
    currentSettings.filterLibrary = libraryEl.value;
    saveSettings(currentSettings);
    // Re-fetch the model list with the updated library filter.
    void refreshModelList(modelSearchEl.value);
  });
}

function setLocalFileStatus(text: string, durationMs = 2500): void {
  localFileDrop.textContent = text;
  setTimeout(() => {
    localFileDrop.textContent = localFileDropDefaultText;
  }, durationMs);
}

function renderLocalFileList(): void {
  const files = listStoredFiles();
  localFilesList.innerHTML = '';

  if (files.length === 0) {
    const li = document.createElement('li');
    li.className = 'local-file-item';
    li.textContent = 'No local files stored yet.';
    localFilesList.appendChild(li);
    return;
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.className = 'local-file-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'local-file-name';
    nameSpan.textContent = file.name;

    const actions = document.createElement('span');
    actions.className = 'local-file-actions';

    const downloadBtn = document.createElement('button');
    downloadBtn.type = 'button';
    downloadBtn.textContent = 'Download';
    downloadBtn.addEventListener('click', async () => {
      const data = await getStoredFile(file.hash);
      if (!data) {
        setLocalFileStatus('File not available in OPFS.', 3000);
        return;
      }
      const blob = new Blob([data]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name || file.hash;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    });

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', async () => {
      await deleteStoredFile(file.hash);
      renderLocalFileList();
      setLocalFileStatus('Removed from local storage.', 2000);
    });

    actions.appendChild(downloadBtn);
    actions.appendChild(deleteBtn);

    li.appendChild(nameSpan);
    li.appendChild(actions);
    localFilesList.appendChild(li);
  }
}

async function handleLocalFile(file: File): Promise<void> {
  try {
    const meta = await storeFile(file, file.name);
    renderLocalFileList();
    setLocalFileStatus(`Saved ${meta.name}`);
  } catch (err) {
    setLocalFileStatus(`Error saving file: ${String(err)}`);
  }
}

export async function handleLocalUrl(rawUrl: string): Promise<void> {
  const url = rawUrl.trim();
  if (!url) return;
  try {
    const meta = await storeFromUrl(url);
    renderLocalFileList();
    setLocalFileStatus(`Fetched and saved ${meta.name}`);
  } catch (err) {
    setLocalFileStatus(`Error fetching URL: ${String(err)}`);
  }
}

export function initLocalFileStore(): void {
  if (!isOpfsAvailable()) {
    // Not supported in this environment; show a hint.
    setLocalFileStatus('OPFS not supported in this browser.');
  }

  renderLocalFileList();

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', async () => {
    if (fileInput.files?.length) {
      await handleLocalFile(fileInput.files[0]);
    }
  });
  document.body.appendChild(fileInput);

  localFileDrop.addEventListener('click', () => fileInput.click());

  localFileDrop.addEventListener('dragover', (e) => {
    e.preventDefault();
    localFileDrop.classList.add('dragover');
  });
  localFileDrop.addEventListener('dragleave', () => {
    localFileDrop.classList.remove('dragover');
  });
  localFileDrop.addEventListener('drop', async (e) => {
    e.preventDefault();
    localFileDrop.classList.remove('dragover');
    if (e.dataTransfer?.files?.length) {
      await handleLocalFile(e.dataTransfer.files[0]);
    }
  });

  localFileAddBtn.addEventListener('click', async () => {
    await handleLocalUrl(localFileUrl.value);
  });
}

function triggerLoad(): void {
  const customId = modelCustomIdEl.value.trim();
  const modelId = customId || selectedModelId;
  if (!modelId) return;
  startWithModel(modelId);
}

// ─── Worker bootstrap ──────────────────────────────────────────────────────────

/**
 * Transition from the model-picker phase to the loading-progress phase and
 * spin up the inference worker for the given model ID.
 */
export function startWithModel(modelId: string): void {
  selectedModelId = modelId;

  // Switch load screen from picker → progress.
  modelPickerEl.classList.add('hidden');
  loadProgressEl.classList.remove('hidden');

  initWorker(modelId);
}

export function initWorker(modelId: string): void {
  const workerUrl =
    (globalThis as any).__Q2_WORKER_URL__ ??
    new URL('./worker.js', import.meta.url).toString();

  worker = new Worker(workerUrl, {
    type: 'module',
  });

  worker.addEventListener('message', (e: MessageEvent<WorkerOutMsg>) => {
    handleWorkerMessage(e.data);
  });

  worker.addEventListener('error', (e) => {
    showError(`Worker error: ${e.message}`);
  });

  const loadMsg: WorkerInMsg = currentSettings.apiToken
    ? { type: 'load', modelId, dtype: currentSettings.dtype, apiToken: currentSettings.apiToken }
    : { type: 'load', modelId, dtype: currentSettings.dtype };
  postToWorker(loadMsg);
}

function postToWorker(msg: WorkerInMsg): void {
  worker?.postMessage(msg);
}

// ─── Worker message handler ────────────────────────────────────────────────────

export function handleWorkerMessage(msg: WorkerOutMsg): void {
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

export function onStatus(
  status: 'loading' | 'ready' | 'generating' | 'idle',
  detail?: string,
): void {
  if (status === 'ready') {
    modelReady = true;
    loadScreen.classList.add('hidden');
    chatApp.classList.remove('hidden');

    // Update model name in the header and sidebar.
    const displayName = selectedModelId.split('/').at(-1) ?? selectedModelId;
    headerTitleEl.textContent = `${displayName} · ${currentSettings.dtype.toUpperCase()} ONNX`;
    sidebarModelTagEl.textContent = displayName;

    inputEl.focus();
  } else if (status === 'loading') {
    loadStatus.textContent = detail ?? 'Loading model…';
  }
}

export function onProgress(file: string, loaded: number, total: number): void {
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

export function onToken(token: string): void {
  if (!activeBubble) return;
  activeRawText += token;
  scheduleBubbleRender();
}

export function onEmbedding(msg: EmbeddingMsg): void {
  embeddingPanel.classList.remove('hidden');

  const { seqLen, hiddenDim, dtype } = msg;
  const expectedElements = seqLen * hiddenDim;

  // Always render the raw activation heat-map for visual feedback.
  // For fp32 data, we can render immediately. For other dtypes, we skip
  // Float32-based stats/heatmaps unless/until we add explicit decoding.
  let floats: Float32Array | null = null;

  if (dtype === 'fp32') {
    if (msg.data.byteLength % 4 !== 0) {
      console.warn(
        `Embedding fp32 data has byteLength=${msg.data.byteLength}, which is not a multiple of 4; skipping Float32 view.`,
      );
    } else {
      const view = new Float32Array(msg.data);
      if (view.length !== expectedElements) {
        console.warn(
          `Embedding fp32 data has length=${view.length}, expected=${expectedElements} (seqLen=${seqLen}, hiddenDim=${hiddenDim}); skipping Float32 view.`,
        );
      } else {
        floats = view;
      }
    }
  }

  if (floats) {
    renderEmbeddingHeatmap(floats, seqLen, hiddenDim);
    embeddingStats.textContent =
      `Shape: [${seqLen} × ${hiddenDim}]  dtype=${dtype}  ` +
      `min=${min(floats).toFixed(3)}  max=${max(floats).toFixed(3)}`;
  } else {
    // Non-fp32 or invalid buffer; render basic shape/dtype info only.
    embeddingStats.textContent =
      `Shape: [${seqLen} × ${hiddenDim}]  dtype=${dtype}  stats=unavailable`;
  }

  // ── Q² kernel ────────────────────────────────────────────────────────────
  // Run the quaternary quantisation in the background.  The WASM kernel is
  // preferred; if instantiation fails (e.g. in test environments that lack
  // WebAssembly.instantiate) we fall back to the pure-TS implementation.
  const n = hiddenDim;
  const dtypeId = DTYPE_TO_Q2[dtype] ?? Q2_DTYPE_FP32;

  void (async () => {
    try {
      const kernel = await getKernel();
      const mem = new Uint8Array(kernel.memory.buffer);

      // Copy the raw activation buffer into WASM memory at the input offset.
      const inputBytes = new Uint8Array(msg.data);
      mem.set(inputBytes, Q2_INPUT_OFFSET);

      // Run quantisation: mean-pool, L2-normalise, threshold, Gray-encode.
      kernel.quantise(Q2_INPUT_OFFSET, seqLen, n, dtypeId, Q2_OUTPUT_OFFSET);

      // Derive the 64-bit transition key.
      const rawKey = kernel.key(Q2_OUTPUT_OFFSET, n);
      const key = BigInt.asUintN(64, rawKey);

      // Read back packed bytes.
      const packed = new Uint8Array(kernel.memory.buffer, Q2_OUTPUT_OFFSET, n >> 2);
      renderQ2Result(packed, key, n);
    } catch {
      // WASM unavailable — use the pure-TypeScript fallback (fp32 only).
      // This path is taken in test environments and SSR contexts.
      // For sub-fp32 dtypes the WASM kernel is required; log a warning and skip.
      if (dtype !== 'fp32') {
        console.warn(`Q² TS fallback: dtype=${dtype} requires WASM kernel; skipping.`);
        return;
      }
      const vec = meanPoolAndNormalise(
        new Float32Array(msg.data),
        seqLen,
        n,
      );
      const { packed, key } = q2EncodeDirect(vec, n);
      renderQ2Result(packed, BigInt.asUintN(64, key), n);
    }
  })();
}

export function onDone(): void {
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

export function sendMessage(): void {
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

export function stopGeneration(): void {
  postToWorker({ type: 'abort' });
}

export function readConfig(): GenerationConfig {
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

export function appendAssistantBubble(): HTMLDivElement {
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
export function renderBubble(bubble: HTMLDivElement, raw: string): void {
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
      const div = document.createElement('div');
      div.innerHTML = escapeAndFormatText(part.text);
      bubble.appendChild(div);
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

export function stripThinkTags(raw: string): string {
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
export function renderEmbeddingHeatmap(
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

export function min(arr: Float32Array): number {
  let m = Infinity;
  for (const v of arr) if (v < m) m = v;
  return m;
}

export function max(arr: Float32Array): number {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  return m;
}

/**
 * Renders the Q² quantisation result in the embedding stats panel.
 *
 * @param packed - n/4 packed Gray-encoded bytes from q2_quantise / q2EncodeDirect
 * @param key    - 64-bit MSB-aligned transition key from q2_key / q2KeyDirect
 * @param n      - original embedding dimension
 */
export function renderQ2Result(packed: Uint8Array, key: bigint, n: number): void {
  // Hex dump of the first 8 bytes (32 symbols) for display.
  const hexBytes = Array.from(packed.slice(0, 8))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  const ellipsis = packed.length > 8 ? '…' : '';
  // Display the key as a zero-padded 16-hex-digit unsigned 64-bit value.
  const keyHex = key.toString(16).padStart(16, '0');
  embeddingStats.textContent +=
    `\nQ²: [${hexBytes}${ellipsis}] (${n >> 2} bytes, ${n} dims)  key=0x${keyHex}`;
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

if (!(globalThis as any).__Q2_SKIP_AUTO_INIT__) {
  initModelPicker();
}

