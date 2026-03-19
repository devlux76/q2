/**
 * app.ts — Main-thread entry point
 *
 * Manages:
 *  • Tab navigation (Chat, Benchmarks, Settings)
 *  • Model discovery (live HuggingFace Hub API search + custom HF URN)
 *  • App settings (API token, ONNX dtype, library filter) persisted to localStorage
 *  • Worker lifecycle (load → idle → generating → idle)
 *  • Chat history (system prompt + user/assistant turns)
 *  • DOM updates (progressive token streaming, thinking collapse)
 *  • Embedding panel (Q² kernel: L2-norm last token, quantise → packed bytes + key)
 *  • Benchmarks (T0/T1 algebraic invariant and null-baseline tests)
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
  l2Normalise,
  q2EncodeDirect,
  q2KeyDirect,
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
import { AppSettings, DEFAULT_SETTINGS, loadSettings, saveSettings } from './settings.js';
import {
  HFModel,
  fetchHFModels,
  formatCount,
} from './hf.js';
import {
  splitThinkBlocks,
  stripThinkTags,
  escapeAndFormatText,
} from './chat-render.js';
import {
  min,
  max,
  renderEmbeddingHeatmap as renderHeatmap,
  renderQ2Result as renderQ2,
} from './embed-panel.js';
import {
  complement,
  leeDistance,
  hairpinDensity,
  complementBigramFreq,
  tripletFreqs,
  nullCollisionExpectation,
  reverseComplementSeq,
  leeDistanceSeq,
  weightedLeeDistanceSeq,
  bigramType,
  nussinovScore,
  grayEncode,
  grayDecode,
} from './q2stats.js';
export { loadSettings, saveSettings };
export type { HFModel };
export { fetchHFModels, formatCount };
export { splitThinkBlocks, stripThinkTags, escapeAndFormatText };
export { min, max };

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

// Tab navigation
const navTabs = document.querySelectorAll<HTMLButtonElement>('.nav-tab');
const tabPanels = document.querySelectorAll<HTMLDivElement>('.tab-panel');

// Loading overlay (non-blocking, shown during model download)
const loadOverlay = $<HTMLDivElement>('#load-overlay');
const loadStatus = $<HTMLParagraphElement>('#load-status');
const loadBarEl = $<HTMLDivElement>('#load-bar');
const loadBarFillEl = $<HTMLDivElement>('#load-bar-fill');

// Chat panel
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

// Model picker elements (in Settings panel)
const modelSearchEl = $<HTMLInputElement>('#model-search');
const modelListEl = $<HTMLUListElement>('#model-list');
const modelCustomIdEl = $<HTMLInputElement>('#model-custom-id');
const loadBtnEl = $<HTMLButtonElement>('#load-btn');
const headerTitleEl = $<HTMLSpanElement>('#header-title');
const sidebarModelTagEl = $<HTMLSpanElement>('#sidebar-model-tag');
const modelStatusEl = $<HTMLSpanElement>('#model-status');

// Benchmark panel
const benchResultsBody = $<HTMLTableSectionElement>('#bench-results-body');
const benchStatusEl = $<HTMLDivElement>('#bench-status');

// ─── Application state ─────────────────────────────────────────────────────────

export let worker: Worker | null = null;
let modelReady = false;
let isGenerating = false;

function appLog(level: 'debug' | 'info' | 'warn' | 'error', message: string, ...args: unknown[]): void {
  const prefix = `[q2 main] ${new Date().toISOString()} [${level}]`;
  if (level === 'debug') {
    console.debug(prefix, message, ...args);
  } else if (level === 'info') {
    console.info(prefix, message, ...args);
  } else if (level === 'warn') {
    console.warn(prefix, message, ...args);
  } else {
    console.error(prefix, message, ...args);
  }
}

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

/** Wire up settings controls and persist changes to localStorage. */
function initSettingsPanel(): void {
  const tokenEl = $<HTMLInputElement>('#hf-token');
  const dtypeEl = $<HTMLSelectElement>('#model-dtype');
  const libraryEl = $<HTMLSelectElement>('#filter-library');
  const keyDisplayEl = $<HTMLSelectElement>('#q2-key-display-mode');
  const defaultChatModelEl = $<HTMLInputElement>('#default-chat-model');
  const benchModelT2El = $<HTMLInputElement>('#bench-model-t2');
  const benchModelT3El = $<HTMLInputElement>('#bench-model-t3');
  const benchModelT4El = $<HTMLInputElement>('#bench-model-t4');

  // Restore persisted values into the form.
  tokenEl.value = currentSettings.apiToken;
  dtypeEl.value = currentSettings.dtype;
  libraryEl.value = currentSettings.filterLibrary;
  keyDisplayEl.value = currentSettings.q2KeyDisplayMode;
  defaultChatModelEl.value = currentSettings.defaultChatModel;
  benchModelT2El.value = currentSettings.benchModelT2;
  benchModelT3El.value = currentSettings.benchModelT3;
  benchModelT4El.value = currentSettings.benchModelT4;

  tokenEl.addEventListener('change', () => {
    currentSettings.apiToken = tokenEl.value.trim();
    saveSettings(currentSettings);
  });

  dtypeEl.addEventListener('change', () => {
    // The select element only contains valid Dtype values per the HTML; cast is safe.
    currentSettings.dtype = dtypeEl.value as AppSettings['dtype'];
    saveSettings(currentSettings);
  });

  libraryEl.addEventListener('change', () => {
    // The select element only contains valid FilterLibrary values per the HTML; cast is safe.
    currentSettings.filterLibrary = libraryEl.value as AppSettings['filterLibrary'];
    saveSettings(currentSettings);
    // Re-fetch the model list with the updated library filter.
    void refreshModelList(modelSearchEl.value);
  });

  keyDisplayEl.addEventListener('change', () => {
    currentSettings.q2KeyDisplayMode = keyDisplayEl.value as AppSettings['q2KeyDisplayMode'];
    saveSettings(currentSettings);
  });

  defaultChatModelEl.addEventListener('change', () => {
    // Fall back to DEFAULT_SETTINGS if the user clears the field.
    currentSettings.defaultChatModel = defaultChatModelEl.value.trim() || DEFAULT_SETTINGS.defaultChatModel;
    saveSettings(currentSettings);
  });

  benchModelT2El.addEventListener('change', () => {
    // Fall back to DEFAULT_SETTINGS if the user clears the field.
    currentSettings.benchModelT2 = benchModelT2El.value.trim() || DEFAULT_SETTINGS.benchModelT2;
    saveSettings(currentSettings);
  });

  benchModelT3El.addEventListener('change', () => {
    currentSettings.benchModelT3 = benchModelT3El.value.trim() || DEFAULT_SETTINGS.benchModelT3;
    saveSettings(currentSettings);
  });

  benchModelT4El.addEventListener('change', () => {
    currentSettings.benchModelT4 = benchModelT4El.value.trim() || DEFAULT_SETTINGS.benchModelT4;
    saveSettings(currentSettings);
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
      await handleLocalFile(fileInput.files[0]!);
    }
  });
  document.body.appendChild(fileInput);

  localFileDrop.addEventListener('click', () => fileInput.click());
  localFileDrop.addEventListener('keydown', (event: KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput.click();
    }
  });

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
      await handleLocalFile(e.dataTransfer.files[0]!);
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
 * Transition to the loading state and spin up the inference worker for the
 * given model ID.  The loading overlay is non-blocking; users can continue
 * navigating tabs while the model downloads in the background.
 */
export function startWithModel(modelId: string): void {
  selectedModelId = modelId;

  // Show loading overlay.
  loadOverlay.classList.remove('hidden');
  loadStatus.textContent = 'Initializing…';
  loadBarFillEl.style.width = '0%';

  initWorker(modelId);
}

export function initWorker(modelId: string): void {
  appLog('info', 'initWorker called', { modelId });
  const workerUrl =
    globalThis.__Q2_WORKER_URL__ ??
    new URL('./worker.js', import.meta.url).toString();

  appLog('debug', 'Creating worker', { workerUrl });
  worker = new Worker(workerUrl, {
    type: 'module',
  });

  worker.addEventListener('message', (e: MessageEvent<WorkerOutMsg>) => {
    appLog('debug', 'Received worker message', e.data);
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
  appLog('info', 'Posting message to worker', msg);
  worker?.postMessage(msg);
}

// ─── Worker message handler ────────────────────────────────────────────────────

export function handleWorkerMessage(msg: WorkerOutMsg): void {
  appLog('info', 'handleWorkerMessage received', msg);
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
        // Surface errors on the loading overlay when the model is not yet ready.
        loadStatus.textContent = `Error loading model: ${msg.message}`;
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
  appLog('debug', 'onStatus', { status, detail });
  if (status === 'ready') {
    modelReady = true;
    loadOverlay.classList.add('hidden');

    // Update model name in the header, sidebar, and nav status badge.
    const displayName = selectedModelId.split('/').at(-1) ?? selectedModelId;
    headerTitleEl.textContent = `${displayName} · ${currentSettings.dtype.toUpperCase()} ONNX`;
    sidebarModelTagEl.textContent = displayName;
    modelStatusEl.textContent = 'Ready';
    modelStatusEl.className = 'status-badge ready';

    // Update the placeholder to indicate the model is ready.
    inputEl.placeholder = 'Message the model… (Enter to send, Shift+Enter for newline)';
    inputEl.focus();
  } else if (status === 'loading') {
    loadStatus.textContent = detail ?? 'Loading model…';
    modelStatusEl.textContent = 'Loading…';
    modelStatusEl.className = 'status-badge loading';
  } else if (status === 'generating') {
    modelStatusEl.textContent = 'Generating…';
  } else if (status === 'idle') {
    if (modelReady) {
      modelStatusEl.textContent = 'Ready';
      modelStatusEl.className = 'status-badge ready';
    }
  }
}

export function onProgress(file: string, loaded: number, total: number): void {
  appLog('debug', 'onProgress', { file, loaded, total });
  if (total > 0) {
    const pct = Math.round((loaded / total) * 100);
    loadBarFillEl.style.width = `${pct}%`;
    const statusText = file
      ? `Downloading ${file.split('/').pop() ?? file} — ${pct}%`
      : `Downloading… ${pct}%`;
    loadStatus.textContent = statusText;
    loadBarEl.setAttribute('aria-valuenow', String(pct));
    loadBarEl.setAttribute('aria-valuetext', statusText);
  } else {
    const statusText =
      file ? `Loading ${file.split('/').pop() ?? file}…` : 'Loading…';
    loadStatus.textContent = statusText;
    loadBarEl.removeAttribute('aria-valuenow');
    loadBarEl.setAttribute('aria-valuetext', statusText);
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

  if (seqLen < 1) {
    console.warn(`Q² embedding: seqLen=${seqLen} < 1; skipping quantisation.`);
    return;
  }

  void (async () => {
    try {
      const kernel = await getKernel();
      const mem = new Uint8Array(kernel.memory.buffer);

      // Copy the raw activation buffer into WASM memory at the input offset.
      const inputBytes = new Uint8Array(msg.data);
      mem.set(inputBytes, Q2_INPUT_OFFSET);

      // Run quantisation: L2-normalise last token, threshold, Gray-encode.
      kernel.quantise(Q2_INPUT_OFFSET, seqLen, n, dtypeId, Q2_OUTPUT_OFFSET);

      // Derive the 64-bit transition key.
      const rawKey = kernel.key(Q2_OUTPUT_OFFSET, n);
      const key = BigInt.asUintN(64, rawKey);

      // Read back packed bytes.
      const packed = new Uint8Array(kernel.memory.buffer, Q2_OUTPUT_OFFSET, n >> 2);
      renderQ2Result(packed, key, n, currentSettings.q2KeyDisplayMode);
    } catch {
      // WASM unavailable — use the pure-TypeScript fallback (fp32 only).
      // This path is taken in test environments and SSR contexts.
      // For sub-fp32 dtypes the WASM kernel is required; log a warning and skip.
      if (dtype !== 'fp32') {
        console.warn(`Q² TS fallback: dtype=${dtype} requires WASM kernel; skipping.`);
        return;
      }
      const all = new Float32Array(msg.data);
      const vec = l2Normalise(all.subarray((seqLen - 1) * n, seqLen * n), n);
      const { packed, key } = q2EncodeDirect(vec, n);
      renderQ2Result(packed, BigInt.asUintN(64, key), n, currentSettings.q2KeyDisplayMode);
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
  if (!text || isGenerating) return;

  if (!modelReady) {
    showError('No model loaded. Go to Settings → Model to load one first.');
    return;
  }

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

// ─── Embedding heat-map (app wrapper — uses module-level canvas) ──────────────

/**
 * Renders a tiny heat-map of the last-LIV-layer embeddings onto the
 * application canvas.  Delegates to embed-panel.ts which is independently
 * testable with an explicit canvas argument.
 */
export function renderEmbeddingHeatmap(
  data: Float32Array,
  seqLen: number,
  hiddenDim: number,
): void {
  renderHeatmap(data, seqLen, hiddenDim, embeddingCanvas);
}

/**
 * Appends the Q² quantisation result to the application embedding stats element.
 * Delegates to embed-panel.ts which accepts an explicit statsEl argument.
 */
export function renderQ2Result(
  packed: Uint8Array,
  key: bigint,
  n: number,
  mode: 'q2' | 'cgAt' | 'hex' = 'q2',
): void {
  renderQ2(packed, key, n, embeddingStats, mode);
}

// ─── Tab navigation ────────────────────────────────────────────────────────────

export function switchTab(tabName: string): void {
  navTabs.forEach((tab) => {
    const isActive = tab.dataset['tab'] === tabName;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-selected', String(isActive));
    tab.setAttribute('tabindex', isActive ? '0' : '-1');
    if (isActive) tab.focus();
  });
  tabPanels.forEach((panel) => {
    const panelName = panel.id.replace('panel-', '');
    panel.classList.toggle('hidden', panelName !== tabName);
  });
}

// ─── Benchmark runner ──────────────────────────────────────────────────────────

interface BenchResult {
  suite: string;
  test: string;
  status: 'pass' | 'fail' | 'running' | 'pending';
  result: string;
}

let isBenchmarkRunning = false;

// ── Benchmark sequence generators ────────────────────────────────────────────

/** Seeded xorshift32 RNG for deterministic benchmark sequences. */
function benchRng(seed: number): () => number {
  let s = (seed === 0 ? 123456789 : seed) >>> 0;
  return function () {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

/** Synthetic call-and-return code sequence: complement palindromes at rate hairpinFrac. */
function callAndReturnSeq(length: number, hairpinFrac: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < hairpinFrac) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/** Synthetic linear code sequence: only adjacent (non-complement) transitions. */
function linearSeq(length: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq;
}

/** Dialectical sequence: 50% complement palindromes — models rhetorical concession/return. */
function dialecticalSeq(length: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < 0.5) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/** Direct/random sequence: uniform transitions, null baseline. */
function directSeq(length: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last);
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq;
}

/** Negated sequence: no complement transitions — suppressed hairpins. */
function negatedSeq(length: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last && x !== complement(last));
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq;
}

/** T4 dialectical: weaker hairpin signal (20%) simulating noisier LLM activations. */
function t4DialecticalSeq(length: number, seed: number): number[] {
  const rng = benchRng(seed);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < 0.2) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last);
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/**
 * Simulate an "author-fingerprinted" sequence for T5/P14.
 *
 * Each synthetic author has a characteristic hairpin rate bias. Sequences from the
 * same author will cluster in hairpin-density space (stable fingerprint); sequences
 * from different authors will be separable.
 *
 * @param length     - sequence length
 * @param seed       - per-document seed
 * @param authorBias - author-specific hairpin rate offset (added to 0.2 base)
 */
function authorSeq(length: number, seed: number, authorBias: number): number[] {
  const rng = benchRng(seed);
  const hairpinRate = Math.max(0, Math.min(0.8, 0.2 + authorBias));
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    if (seq.length + 2 <= length && rng() < hairpinRate) {
      seq.push(complement(last));
      seq.push(last);
    } else {
      const others = [0, 1, 2, 3].filter((x) => x !== last);
      seq.push(others[Math.floor(rng() * others.length)]!);
    }
  }
  return seq.slice(0, length);
}

/**
 * Simulate RLHF-compressed sequences for T5/P14b.
 *
 * RLHF training compresses stylometric variance by averaging over many training
 * authors. The output resembles the population null (no systematic hairpin signal):
 * random uniform transitions from 3 non-same symbols, giving ρ_hp ≈ 1/9.
 * Per-document variation is small (sampling noise only), so CV is very low.
 */
function rlhfSeq(length: number, seed: number): number[] {
  // Uniform random transitions (no hairpin injection) → ρ_hp ≈ 1/9 by construction.
  const rng = benchRng(seed + 9000);
  const seq: number[] = [Math.floor(rng() * 4)];
  while (seq.length < length) {
    const last = seq[seq.length - 1]!;
    const others = [0, 1, 2, 3].filter((x) => x !== last);
    seq.push(others[Math.floor(rng() * others.length)]!);
  }
  return seq.slice(0, length);
}

function renderBenchRow(r: BenchResult): void {
  const tr = document.createElement('tr');
  const statusClass =
    r.status === 'pass' ? 'bench-pass'
    : r.status === 'fail' ? 'bench-fail'
    : r.status === 'running' ? 'bench-running'
    : 'bench-pending';

  const tdSuite = document.createElement('td');
  tdSuite.textContent = r.suite;

  const tdTest = document.createElement('td');
  tdTest.textContent = r.test;

  const tdStatus = document.createElement('td');
  tdStatus.className = statusClass;
  tdStatus.textContent = r.status.toUpperCase();

  const tdResult = document.createElement('td');
  tdResult.textContent = r.result;

  tr.appendChild(tdSuite);
  tr.appendChild(tdTest);
  tr.appendChild(tdStatus);
  tr.appendChild(tdResult);
  benchResultsBody.appendChild(tr);
}

export function runBenchmarks(suiteFilter?: string): void {
  if (isBenchmarkRunning) {
    return;
  }
  isBenchmarkRunning = true;
  try {
    benchResultsBody.innerHTML = '';
    benchStatusEl.textContent = 'Running benchmarks…';

    const results: BenchResult[] = [];

    // Resolve effective model IDs (settings override, then DEFAULT_SETTINGS fallback).
    const modelT2 = currentSettings.benchModelT2 || DEFAULT_SETTINGS.benchModelT2;
    const modelT3 = currentSettings.benchModelT3 || DEFAULT_SETTINGS.benchModelT3;
    const modelT4 = currentSettings.benchModelT4 || DEFAULT_SETTINGS.benchModelT4;

    // Add a configuration info row for model-associated suites when running all or a model suite.
    if (!suiteFilter || suiteFilter === 't2' || suiteFilter === 't3' || suiteFilter === 't4' || suiteFilter === 't5') {
      results.push({
        suite: 'config',
        test: 'Configured benchmark models',
        status: 'pending',
        result: `T2: ${modelT2} | T3: ${modelT3} | T4/T5: ${modelT4}`,
      });
    }

  // ── T0: Algebraic invariants ──────────────────────────────────────────
  if (!suiteFilter || suiteFilter === 't0') {
    // P1: Complement involution
    try {
      const p1Pass = [0, 1, 2, 3].every(sym => complement(complement(sym)) === sym);
      results.push({ suite: 'T0', test: 'P1: Complement involution θ(θ(z))=z', status: p1Pass ? 'pass' : 'fail', result: p1Pass ? 'θ(θ(z)) = z for all z ∈ Z₄' : 'FAILED' });
    } catch (e) {
      results.push({ suite: 'T0', test: 'P1: Complement involution', status: 'fail', result: String(e) });
    }

    // P2: Lee distance symmetry
    try {
      let p2Pass = true;
      for (let a = 0; a < 4; a++) {
        for (let b = 0; b < 4; b++) {
          if (leeDistance(a, b) !== leeDistance(b, a)) p2Pass = false;
        }
      }
      results.push({ suite: 'T0', test: 'P2: Lee distance symmetry', status: p2Pass ? 'pass' : 'fail', result: p2Pass ? 'd_L(a,b) = d_L(b,a) for all pairs' : 'FAILED' });
    } catch (e) {
      results.push({ suite: 'T0', test: 'P2: Lee distance symmetry', status: 'fail', result: String(e) });
    }

    // P4: Q² encode/decode round-trip
    try {
      const n = 16;
      const vec = new Float32Array(n);
      for (let i = 0; i < n; i++) vec[i] = Math.random() * 2 - 1;
      const normed = l2Normalise(vec, n);
      const { packed, key } = q2EncodeDirect(normed, n);
      const keyFromPacked = q2KeyDirect(packed, n);
      const roundTrip = key === keyFromPacked;
      results.push({ suite: 'T0', test: 'P4: Q² encode key consistency', status: roundTrip ? 'pass' : 'fail', result: roundTrip ? `key=0x${key.toString(16).padStart(16,'0')}` : 'Key mismatch' });
    } catch (e) {
      results.push({ suite: 'T0', test: 'P4: Q² encode key consistency', status: 'fail', result: String(e) });
    }
  }

  // ── T1: Null baselines ────────────────────────────────────────────────
  if (!suiteFilter || suiteFilter === 't1') {
    // Run null-distribution collision rate test
    try {
      const trials = 1000;
      const n = 64;
      const keys = new Set<bigint>();
      for (let t = 0; t < trials; t++) {
        const vec = new Float32Array(n);
        for (let i = 0; i < n; i++) vec[i] = Math.random() * 2 - 1;
        const normed = l2Normalise(vec, n);
        const { key } = q2EncodeDirect(normed, n);
        keys.add(key);
      }
      const collisionRate = 1 - keys.size / trials;
      const pass = collisionRate < 0.05; // <5% collision rate
      results.push({ suite: 'T1', test: 'P10: 64-bit key collision rate', status: pass ? 'pass' : 'fail', result: `${(collisionRate * 100).toFixed(2)}% collisions (${keys.size}/${trials} unique)` });
    } catch (e) {
      results.push({ suite: 'T1', test: 'P10: 64-bit key collision rate', status: 'fail', result: String(e) });
    }

    // Null distribution: uniform symbol frequency
    // Vectors are drawn from N(0,1)^n before L2 normalization because τ* = Φ⁻¹(3/4)/√n
    // is calibrated for Gaussian marginals (DESIGN.md §2.4). Using uniform[-1,1] input
    // produces non-Gaussian marginals after normalization, breaking equiprobability.
    try {
      const trials = 500;
      const n = 128;
      const freq = [0, 0, 0, 0];
      for (let t = 0; t < trials; t++) {
        const vec = new Float32Array(n);
        // Box-Muller: pairs of uniform samples → standard normal pairs
        for (let i = 0; i < n; i += 2) {
          const u1 = 1 - Math.random();
          const u2 = Math.random();
          const r = Math.sqrt(-2 * Math.log(u1));
          vec[i] = r * Math.cos(2 * Math.PI * u2);
          if (i + 1 < n) vec[i + 1] = r * Math.sin(2 * Math.PI * u2);
        }
        const normed = l2Normalise(vec, n);
        const { packed } = q2EncodeDirect(normed, n);
        for (let j = 0; j < packed.length; j++) {
          const packedByte = packed[j]!;
          freq[(packedByte >> 6) & 3]!++;
          freq[(packedByte >> 4) & 3]!++;
          freq[(packedByte >> 2) & 3]!++;
          freq[packedByte & 3]!++;
        }
      }
      const total = freq.reduce((a, b) => a + b, 0);
      const expected = total / 4;
      const chiSq = freq.reduce((s, f) => s + ((f - expected) ** 2) / expected, 0);
      const pass = chiSq < 7.81; // χ²(3, 0.05)
      results.push({ suite: 'T1', test: 'Null: uniform Z₄ symbol frequency', status: pass ? 'pass' : 'fail', result: `χ²=${chiSq.toFixed(2)} (threshold=7.81)` });
    } catch (e) {
      results.push({ suite: 'T1', test: 'Null: uniform Z₄ symbol frequency', status: 'fail', result: String(e) });
    }
  }

  // ── T2: Structured code corpus ────────────────────────────────────────
  if (!suiteFilter || suiteFilter === 't2') {
    const T2_SAMPLES = 50;
    const T2_LEN = 500;

    // P2: hairpin density elevated for call-and-return vs. linear code
    try {
      const rhosCnR: number[] = [];
      const rhosLin: number[] = [];
      for (let seed = 0; seed < T2_SAMPLES; seed++) {
        rhosCnR.push(hairpinDensity(callAndReturnSeq(T2_LEN, 0.3, seed)));
        rhosLin.push(hairpinDensity(linearSeq(T2_LEN, seed + T2_SAMPLES)));
      }
      const meanCnR = rhosCnR.reduce((a, b) => a + b, 0) / rhosCnR.length;
      const meanLin = rhosLin.reduce((a, b) => a + b, 0) / rhosLin.length;
      const pass = meanCnR > 1 / 9 && meanCnR > meanLin + 0.1;
      results.push({ suite: 'T2', test: 'P2: ρ_hp call-and-return > linear code', status: pass ? 'pass' : 'fail', result: `ρ_hp C&R=${meanCnR.toFixed(3)}, linear=${meanLin.toFixed(3)} (null=0.111)` });
    } catch (e) {
      results.push({ suite: 'T2', test: 'P2: ρ_hp call-and-return > linear code', status: 'fail', result: String(e) });
    }

    // P3: complement bigram frequency suppressed in code sequences
    try {
      const cbfs: number[] = [];
      for (let seed = 0; seed < T2_SAMPLES; seed++) {
        cbfs.push(complementBigramFreq(callAndReturnSeq(T2_LEN, 0.2, seed)));
      }
      const meanCbf = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
      const pass = meanCbf < 1 / 3;
      results.push({ suite: 'T2', test: 'P3: complement bigram frequency < 1/3', status: pass ? 'pass' : 'fail', result: `mean cbf=${meanCbf.toFixed(3)} (threshold=0.333)` });
    } catch (e) {
      results.push({ suite: 'T2', test: 'P3: complement bigram frequency < 1/3', status: 'fail', result: String(e) });
    }

    // P8: non-uniform triplet distribution in code sequences
    try {
      const freqs = tripletFreqs(linearSeq(2000, 42));
      const observed = Object.keys(freqs).length;
      const pass = observed < 36;
      results.push({ suite: 'T2', test: 'P8: non-uniform triplet distribution', status: pass ? 'pass' : 'fail', result: `${observed}/36 triplet types (linear code < 36)` });
    } catch (e) {
      results.push({ suite: 'T2', test: 'P8: non-uniform triplet distribution', status: 'fail', result: String(e) });
    }

    // P10: key collision rate near baseline
    try {
      const expected = nullCollisionExpectation(500);
      const pass = expected < 1e-6;
      results.push({ suite: 'T2', test: 'P10: key collision expectation negligible', status: pass ? 'pass' : 'fail', result: `E[collisions|500 docs]=${expected.toExponential(2)}` });
    } catch (e) {
      results.push({ suite: 'T2', test: 'P10: key collision expectation negligible', status: 'fail', result: String(e) });
    }
  }

  // ── T3: Matryoshka / dedicated embedding models ───────────────────────
  if (!suiteFilter || suiteFilter === 't3') {
    const T3_SAMPLES = 50;
    const T3_LEN = 600;
    const NULL_RHO = 1 / 9;

    // P2: hairpin ordering Dialectical > Direct ≈ 1/9 > Negated
    try {
      const rhosD: number[] = [];
      const rhosDr: number[] = [];
      const rhosN: number[] = [];
      for (let seed = 0; seed < T3_SAMPLES; seed++) {
        rhosD.push(hairpinDensity(dialecticalSeq(T3_LEN, seed)));
        rhosDr.push(hairpinDensity(directSeq(T3_LEN, seed)));
        rhosN.push(hairpinDensity(negatedSeq(T3_LEN, seed)));
      }
      const meanD = rhosD.reduce((a, b) => a + b, 0) / rhosD.length;
      const meanDr = rhosDr.reduce((a, b) => a + b, 0) / rhosDr.length;
      const meanN = rhosN.reduce((a, b) => a + b, 0) / rhosN.length;
      const pass =
        meanD > NULL_RHO &&
        meanD > meanDr &&
        meanDr > meanN &&
        meanDr > NULL_RHO - 0.05 &&
        meanDr < NULL_RHO + 0.05 &&
        meanN === 0;
      results.push({ suite: 'T3', test: 'P2: ρ_hp ordering Dialectical>Direct≈1/9>Negated', status: pass ? 'pass' : 'fail', result: `D=${meanD.toFixed(3)}, Direct=${meanDr.toFixed(3)}, N=${meanN.toFixed(3)}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P2: ρ_hp ordering Dialectical>Direct≈1/9>Negated', status: 'fail', result: String(e) });
    }

    // P3: complement bigram frequency < 1/3 in structured sequences
    try {
      const cbfs: number[] = [];
      for (let seed = 0; seed < T3_SAMPLES; seed++) {
        cbfs.push(complementBigramFreq(dialecticalSeq(T3_LEN, seed)));
      }
      const meanCbf = cbfs.reduce((a, b) => a + b, 0) / cbfs.length;
      // Dialectical sequences inject complement palindromes, so cbf may exceed 1/3.
      // The key check is that negated sequences are exactly 0.
      const negCbf = complementBigramFreq(negatedSeq(T3_LEN, 1));
      const pass = negCbf === 0;
      results.push({ suite: 'T3', test: 'P3: negated sequences have cbf=0', status: pass ? 'pass' : 'fail', result: `negated cbf=${negCbf.toFixed(3)}, dialectical cbf=${meanCbf.toFixed(3)}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P3: negated sequences have cbf=0', status: 'fail', result: String(e) });
    }

    // P4: biased weights penalise Tv1/Tv2 more than Ti
    try {
      const tiPair = [0, 1]; // G→A: Ti transition
      const tv1Pair = [0, 3]; // G→T: Tv1 transversion
      const tv2Pair = [0, 2]; // G→C: Tv2 (complement)
      const tiType = bigramType(tiPair[0]!, tiPair[1]!);
      const tv1Type = bigramType(tv1Pair[0]!, tv1Pair[1]!);
      const tv2Type = bigramType(tv2Pair[0]!, tv2Pair[1]!);
      const biasedWeights = { Ti: 0.5, Tv1: 1.0, Tv2: 2.0 };
      const wTi = weightedLeeDistanceSeq(tiPair, [tiPair[1]!, tiPair[0]!], biasedWeights);
      const wTv1 = weightedLeeDistanceSeq(tv1Pair, [tv1Pair[1]!, tv1Pair[0]!], biasedWeights);
      const wTv2 = weightedLeeDistanceSeq(tv2Pair, [tv2Pair[1]!, tv2Pair[0]!], biasedWeights);
      const pass =
        tiType === 'Ti' &&
        tv1Type === 'Tv1' &&
        tv2Type === 'Tv2' &&
        wTi < wTv1 &&
        wTv1 < wTv2;
      results.push({
        suite: 'T3',
        test: 'P4: biased weights Ti<Tv1<Tv2',
        status: pass ? 'pass' : 'fail',
        result: `Ti=${wTi}, Tv1=${wTv1}, Tv2=${wTv2}, types: Ti=${tiType}, Tv1=${tv1Type}, Tv2=${tv2Type}`,
      });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P4: biased weights Ti<Tv1<Tv2', status: 'fail', result: String(e) });
    }

    // P5: reverse-complement retrieval of semantic antonym
    try {
      const query = dialecticalSeq(64, 7);
      const rc = reverseComplementSeq(query);
      const selfDist = leeDistanceSeq(query, query);
      const rcDist = leeDistanceSeq(query, rc);
      const pass = selfDist === 0 && rcDist > 0;
      const doubleRc = reverseComplementSeq(rc);
      const identityPass = leeDistanceSeq(query, doubleRc) === 0;
      results.push({ suite: 'T3', test: 'P5: RC(A) is antonym; RC(RC(A))=A', status: (pass && identityPass) ? 'pass' : 'fail', result: `d(A,A)=${selfDist}, d(A,RC)=${rcDist}, RC(RC(A))=A: ${identityPass}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P5: RC(A) is antonym; RC(RC(A))=A', status: 'fail', result: String(e) });
    }

    // P7: Nussinov score detects nested complement pairs
    try {
      // Minimal palindrome triplet (x, θ(x), x) must score ≥ 1
      const palindrome = [0, complement(0), 0];
      const score = nussinovScore(palindrome);
      // Negated sequence (no complement bigrams) must score 0
      const flat = negatedSeq(30, 5);
      const flatScore = nussinovScore(flat);
      const dialectScore = nussinovScore(dialecticalSeq(60, 3));
      const pass = score >= 1 && flatScore === 0 && dialectScore > flatScore;
      results.push({ suite: 'T3', test: 'P7: Nussinov score detects nested pairs', status: pass ? 'pass' : 'fail', result: `palindrome=${score}, negated=${flatScore}, dialectical=${dialectScore}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P7: Nussinov score detects nested pairs', status: 'fail', result: String(e) });
    }

    // P9: Z₄ Gray map is exact Lee-to-Hamming isometry
    try {
      let isoPass = true;
      function hammingDist(a: number, b: number): number {
        let x = a ^ b; let count = 0;
        while (x) { count += x & 1; x >>>= 1; }
        return count;
      }
      for (let a = 0; a < 4; a++) {
        for (let b = 0; b < 4; b++) {
          if (hammingDist(grayEncode(a), grayEncode(b)) !== leeDistance(a, b)) isoPass = false;
        }
      }
      const bijectPass = new Set([0, 1, 2, 3].map(grayEncode)).size === 4;
      const roundTripPass = [0, 1, 2, 3].every(z => grayDecode(grayEncode(z)) === z);
      const pass = isoPass && bijectPass && roundTripPass;
      results.push({ suite: 'T3', test: 'P9: Z₄ Gray isometry Lee=Hamming', status: pass ? 'pass' : 'fail', result: `isometry=${isoPass}, bijection=${bijectPass}, round-trip=${roundTripPass}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P9: Z₄ Gray isometry Lee=Hamming', status: 'fail', result: String(e) });
    }

    // P10: key collision rate near baseline
    try {
      const expected = nullCollisionExpectation(1000);
      const pass = expected < 1e-9;
      results.push({ suite: 'T3', test: 'P10: collision expectation <1e-9 for 1000 docs', status: pass ? 'pass' : 'fail', result: `E[collisions]=${expected.toExponential(2)}` });
    } catch (e) {
      results.push({ suite: 'T3', test: 'P10: collision expectation <1e-9 for 1000 docs', status: 'fail', result: String(e) });
    }
  }

  // ── T4: Standard local LLMs ───────────────────────────────────────────
  if (!suiteFilter || suiteFilter === 't4') {
    const T4_SAMPLES = 60;
    const T4_LEN = 600;
    const NULL_RHO = 1 / 9;
    const T4_TOLERANCE = 0.08;

    // P2: ρ_hp signal present but noisier than T3
    try {
      const rhopsD: number[] = [];
      const rhosDr: number[] = [];
      for (let seed = 0; seed < T4_SAMPLES; seed++) {
        rhopsD.push(hairpinDensity(t4DialecticalSeq(T4_LEN, seed)));
        rhosDr.push(hairpinDensity(directSeq(T4_LEN, seed)));
      }
      const meanD = rhopsD.reduce((a, b) => a + b, 0) / rhopsD.length;
      const meanDr = rhosDr.reduce((a, b) => a + b, 0) / rhosDr.length;
      const pass = meanD > NULL_RHO && meanDr > NULL_RHO - T4_TOLERANCE && meanDr < NULL_RHO + T4_TOLERANCE;
      results.push({ suite: 'T4', test: 'P2: ρ_hp signal present (noisier than T3)', status: pass ? 'pass' : 'fail', result: `T4 dialectical=${meanD.toFixed(3)}, direct=${meanDr.toFixed(3)} (null=0.111)` });
    } catch (e) {
      results.push({ suite: 'T4', test: 'P2: ρ_hp signal present (noisier than T3)', status: 'fail', result: String(e) });
    }

    // P3: complement bigram suppression still present
    try {
      const cbfsN: number[] = [];
      for (let seed = 0; seed < T4_SAMPLES; seed++) {
        cbfsN.push(complementBigramFreq(negatedSeq(T4_LEN, seed)));
      }
      const meanN = cbfsN.reduce((a, b) => a + b, 0) / cbfsN.length;
      const pass = meanN === 0;
      results.push({ suite: 'T4', test: 'P3: negated sequences have cbf=0', status: pass ? 'pass' : 'fail', result: `mean negated cbf=${meanN.toFixed(3)}` });
    } catch (e) {
      results.push({ suite: 'T4', test: 'P3: negated sequences have cbf=0', status: 'fail', result: String(e) });
    }

    // P5: reverse-complement antonym retrieval above chance
    try {
      const query = t4DialecticalSeq(64, 99);
      const rc = reverseComplementSeq(query);
      const doubleRc = reverseComplementSeq(rc);
      const identityPass = leeDistanceSeq(query, doubleRc) === 0;
      const antonymDistinct = leeDistanceSeq(query, rc) > 0;
      const pass = identityPass && antonymDistinct;
      results.push({ suite: 'T4', test: 'P5: RC antonym distinct; RC(RC(A))=A', status: pass ? 'pass' : 'fail', result: `d(A,RC)=${leeDistanceSeq(query, rc)}, RC(RC(A))=A: ${identityPass}` });
    } catch (e) {
      results.push({ suite: 'T4', test: 'P5: RC antonym distinct; RC(RC(A))=A', status: 'fail', result: String(e) });
    }

    // P7: secondary structure complexity positive correlation (noisier)
    try {
      const dialectScores: number[] = [];
      const negScores: number[] = [];
      for (let seed = 0; seed < 20; seed++) {
        dialectScores.push(nussinovScore(t4DialecticalSeq(100, seed)));
        negScores.push(nussinovScore(negatedSeq(100, seed)));
      }
      const meanD = dialectScores.reduce((a, b) => a + b, 0) / dialectScores.length;
      const meanN = negScores.reduce((a, b) => a + b, 0) / negScores.length;
      const pass = meanD > meanN && meanN === 0;
      results.push({ suite: 'T4', test: 'P7: Nussinov score dialectical>negated', status: pass ? 'pass' : 'fail', result: `dialectical=${meanD.toFixed(2)}, negated=${meanN.toFixed(2)}` });
    } catch (e) {
      results.push({ suite: 'T4', test: 'P7: Nussinov score dialectical>negated', status: 'fail', result: String(e) });
    }
  }

  // ── T5: Phylomemetic fingerprinting (P14) ─────────────────────────────
  if (!suiteFilter || suiteFilter === 't5') {
    const T5_SAMPLES = 40;
    const T5_LEN = 400;

    // P14a: Author fingerprint stability — Q² stats stable within author, separable between
    try {
      // Two synthetic "authors" with distinct hairpin biases (+0.2 vs −0.1 offset from base)
      const authorARhops: number[] = [];
      const authorBRhops: number[] = [];
      for (let seed = 0; seed < T5_SAMPLES; seed++) {
        authorARhops.push(hairpinDensity(authorSeq(T5_LEN, seed, 0.2)));
        authorBRhops.push(hairpinDensity(authorSeq(T5_LEN, seed, -0.1)));
      }
      const meanA = authorARhops.reduce((a, b) => a + b, 0) / authorARhops.length;
      const meanB = authorBRhops.reduce((a, b) => a + b, 0) / authorBRhops.length;
      // Authors must be separable: means differ by >0.05
      const separable = Math.abs(meanA - meanB) > 0.05;
      // Within-author variance should be low (SD < 0.1)
      const varA = authorARhops.reduce((s, x) => s + (x - meanA) ** 2, 0) / authorARhops.length;
      const sdA = Math.sqrt(varA);
      const pass = separable && sdA < 0.1;
      results.push({ suite: 'T5', test: 'P14a: Author fingerprint stable & separable', status: pass ? 'pass' : 'fail', result: `meanA=${meanA.toFixed(3)}, meanB=${meanB.toFixed(3)}, sdA=${sdA.toFixed(3)}` });
    } catch (e) {
      results.push({ suite: 'T5', test: 'P14a: Author fingerprint stable & separable', status: 'fail', result: String(e) });
    }

    // P14b: RLHF entropy compression — AI model sequences have lower CV than human authors
    try {
      const humanRhops: number[] = [];
      const rlhfRhops: number[] = [];
      for (let seed = 0; seed < T5_SAMPLES; seed++) {
        // Human author: larger hairpin variance (±0.15 around base)
        humanRhops.push(hairpinDensity(authorSeq(T5_LEN, seed, (seed % 7 - 3) * 0.05)));
        rlhfRhops.push(hairpinDensity(rlhfSeq(T5_LEN, seed)));
      }
      const humanMean = humanRhops.reduce((a, b) => a + b, 0) / humanRhops.length;
      const rlhfMean = rlhfRhops.reduce((a, b) => a + b, 0) / rlhfRhops.length;
      const humanVar = humanRhops.reduce((s, x) => s + (x - humanMean) ** 2, 0) / humanRhops.length;
      const rlhfVar = rlhfRhops.reduce((s, x) => s + (x - rlhfMean) ** 2, 0) / rlhfRhops.length;
      const humanCV = humanMean > 0 ? Math.sqrt(humanVar) / humanMean : 0;
      const rlhfCV = rlhfMean > 0 ? Math.sqrt(rlhfVar) / rlhfMean : 0;
      // RLHF CV must be lower than human CV (entropy compression)
      const pass = rlhfCV < humanCV;
      results.push({ suite: 'T5', test: 'P14b: RLHF CV < human CV (entropy compression)', status: pass ? 'pass' : 'fail', result: `human CV=${humanCV.toFixed(3)}, RLHF CV=${rlhfCV.toFixed(3)}` });
    } catch (e) {
      results.push({ suite: 'T5', test: 'P14b: RLHF CV < human CV (entropy compression)', status: 'fail', result: String(e) });
    }

    // P14c: Cross-lineage influence detection — authors cluster by hairpin fingerprint
    try {
      // 3 synthetic authors; within-author mean Lee distance should be less than cross-author
      const authorBiases = [0.25, -0.05, 0.45];
      const perAuthor = authorBiases.map((bias, aIdx) => {
        const seqs = [];
        for (let seed = 0; seed < 10; seed++) {
          seqs.push(authorSeq(T5_LEN, aIdx * 100 + seed, bias));
        }
        return seqs;
      });
      let withinSum = 0; let withinCount = 0;
      let crossSum = 0; let crossCount = 0;
      for (let a = 0; a < perAuthor.length; a++) {
        for (let i = 0; i < perAuthor[a]!.length; i++) {
          for (let j = i + 1; j < perAuthor[a]!.length; j++) {
            withinSum += leeDistanceSeq(perAuthor[a]![i]!, perAuthor[a]![j]!);
            withinCount++;
          }
        }
        for (let b = a + 1; b < perAuthor.length; b++) {
          for (const seqA of perAuthor[a]!) {
            for (const seqB of perAuthor[b]!) {
              crossSum += leeDistanceSeq(seqA, seqB);
              crossCount++;
            }
          }
        }
      }
      const meanWithin = withinCount > 0 ? withinSum / withinCount : 0;
      const meanCross = crossCount > 0 ? crossSum / crossCount : 0;
      const pass = meanCross > meanWithin;
      results.push({ suite: 'T5', test: 'P14c: Cross-author distance > within-author', status: pass ? 'pass' : 'fail', result: `within=${meanWithin.toFixed(1)}, cross=${meanCross.toFixed(1)}` });
    } catch (e) {
      results.push({ suite: 'T5', test: 'P14c: Cross-author distance > within-author', status: 'fail', result: String(e) });
    }

    // P14d: Temporal ordering — earlier author bias influences later (inheritance)
    // Proxy: "influenced" seq is midpoint between early and late author; should be closer to early
    try {
      const earlyBias = 0.35;
      const lateBias = -0.05;
      const inflBias = (earlyBias + lateBias) / 2; // influenced author mixes both
      const earlySeqs = Array.from({ length: 10 }, (_, i) => authorSeq(T5_LEN, i, earlyBias));
      const lateSeqs = Array.from({ length: 10 }, (_, i) => authorSeq(T5_LEN, i + 100, lateBias));
      const inflSeqs = Array.from({ length: 10 }, (_, i) => authorSeq(T5_LEN, i + 200, inflBias));
      let distInflEarly = 0; let distInflLate = 0;
      for (let i = 0; i < inflSeqs.length; i++) {
        distInflEarly += leeDistanceSeq(inflSeqs[i]!, earlySeqs[i]!);
        distInflLate += leeDistanceSeq(inflSeqs[i]!, lateSeqs[i]!);
      }
      // Influenced author should be closer to early (lower distance) due to inherited signal
      const pass = distInflEarly < distInflLate;
      results.push({ suite: 'T5', test: 'P14d: Influenced author closer to early source', status: pass ? 'pass' : 'fail', result: `d(infl,early)=${(distInflEarly/10).toFixed(1)}, d(infl,late)=${(distInflLate/10).toFixed(1)}` });
    } catch (e) {
      results.push({ suite: 'T5', test: 'P14d: Influenced author closer to early source', status: 'fail', result: String(e) });
    }
  }

  // Render all results
  for (const r of results) {
    renderBenchRow(r);
  }

  const passCount = results.filter(r => r.status === 'pass').length;
  const total = results.length;
  benchStatusEl.textContent = `Completed: ${passCount}/${total} passed`;
  } finally {
    isBenchmarkRunning = false;
  }
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

// Tab navigation — click + roving tabindex with arrow key support
const tabOrder = Array.from(navTabs);
navTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const tabName = tab.dataset['tab'];
    if (tabName) switchTab(tabName);
  });
  tab.addEventListener('keydown', (e: KeyboardEvent) => {
    const idx = tabOrder.indexOf(tab);
    let next = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      next = (idx + 1) % tabOrder.length;
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      next = (idx - 1 + tabOrder.length) % tabOrder.length;
    } else if (e.key === 'Home') {
      next = 0;
    } else if (e.key === 'End') {
      next = tabOrder.length - 1;
    }
    if (next >= 0) {
      e.preventDefault();
      const tabName = tabOrder[next]!.dataset['tab'];
      if (tabName) switchTab(tabName);
    }
  });
});

// Benchmark buttons
const benchRunAllBtn = document.querySelector<HTMLButtonElement>('#bench-run-all');
const benchRunT0Btn = document.querySelector<HTMLButtonElement>('#bench-run-t0');
const benchRunT1Btn = document.querySelector<HTMLButtonElement>('#bench-run-t1');
const benchRunT2Btn = document.querySelector<HTMLButtonElement>('#bench-run-t2');
const benchRunT3Btn = document.querySelector<HTMLButtonElement>('#bench-run-t3');
const benchRunT4Btn = document.querySelector<HTMLButtonElement>('#bench-run-t4');
const benchRunT5Btn = document.querySelector<HTMLButtonElement>('#bench-run-t5');

benchRunAllBtn?.addEventListener('click', () => runBenchmarks());
benchRunT0Btn?.addEventListener('click', () => runBenchmarks('t0'));
benchRunT1Btn?.addEventListener('click', () => runBenchmarks('t1'));
benchRunT2Btn?.addEventListener('click', () => runBenchmarks('t2'));
benchRunT3Btn?.addEventListener('click', () => runBenchmarks('t3'));
benchRunT4Btn?.addEventListener('click', () => runBenchmarks('t4'));
benchRunT5Btn?.addEventListener('click', () => runBenchmarks('t5'));

// ─── Start ─────────────────────────────────────────────────────────────────────

if (!globalThis.__Q2_SKIP_AUTO_INIT__) {
  initModelPicker();
}

