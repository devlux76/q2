var __defProp = Object.defineProperty;
var __returnValue = (v) => v;
function __exportSetter(name, newValue) {
  this[name] = __returnValue.bind(null, newValue);
}
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: __exportSetter.bind(all, name)
    });
};
var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
  get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
}) : x)(function(x) {
  if (typeof require !== "undefined")
    return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x + '" is not supported');
});

// src/q2.ts
var Q2_DTYPE_FP32 = 0;
var Q2_DTYPE_FP16 = 1;
var Q2_DTYPE_Q8 = 2;
var Q2_DTYPE_Q4 = 3;
var Q2_DTYPE_Q2 = 4;
var DTYPE_TO_Q2 = {
  fp32: Q2_DTYPE_FP32,
  fp16: Q2_DTYPE_FP16,
  q8: Q2_DTYPE_Q8,
  q4: Q2_DTYPE_Q4,
  q2: Q2_DTYPE_Q2
};
var WASM_B64 = "AGFzbQEAAAABHARgAX8BfWADf39/AX1gBX9/f39/AX9gAn9/AX4DBQQAAQIDBQMBAAgGBgF/AEEACwce" + "AwNtZW0CAAtxMl9xdWFudGlzZQACBnEyX2tleQADCroHBFgBA38gAEGAgAJxQRB0IQEgAEEKdkEfcSEC" + "IABB/wdxIQMgAkUEQCABvg8LIAJBH0YEQCABQYCAgPwHIANBDXRycr4PCyABIAJB8ABqQRd0IANBDXRy" + "cr4LigEBAX8CQAJAAkACQAJAIAIOBAABAgMECyAAIAFBAnRqKgIADwsgACABQQF0ai8BABAADwsgACAB" + "aiwAALIPCyAAIAFBAXZqLQAAIQMgAUEBcQRAIANBD3EhAwUgA0EEdiEDCyADQQhrsg8LIAAgAUECdmot" + "AAAhAyADQQMgAUEDcWtBAXR2QQNxswu7BAQFfwR9BH8BfSACQQJ2IQUgA0EERgRAQQAhBgJAA0AgBiAF" + "Tw0BIAQgBmogACAGai0AADoAACAGQQFqIQYMAAsLIAUPC0EAIQYCQANAIAYgAk8NASMAIAZBAnRqQwAA" + "AAA4AgAgBkEBaiEGDAALC0EAIQcCQANAIAcgAU8NAUEAIQYCQANAIAYgAk8NASAHIAJsIAZqIQgjACAG" + "QQJ0aiEJIAkgCSoCACAAIAggAxABkjgCACAGQQFqIQYMAAsLIAdBAWohBwwACwsgAbMhEkEAIQYCQANA" + "IAYgAk8NASMAIAZBAnRqIQkgCSAJKgIAIBKVOAIAIAZBAWohBgwACwtDAAAAACELQQAhBgJAA0AgBiAC" + "Tw0BIwAgBkECdGoqAgAhCiALIAogCpSSIQsgBkEBaiEGDAALCyALQ5WV5iReBEBDAACAPyALkZUhDEEA" + "IQYCQANAIAYgAk8NASMAIAZBAnRqIQkgCSAJKgIAIAyUOAIAIAZBAWohBgwACwsLQwisLD8gArORlSEN" + "QQAhBgJAA0AgBiAFTw0BIAQgBmpBADoAACAGQQFqIQYMAAsLQQAhBgJAA0AgBiACTw0BIwAgBkECdGoq" + "AgAhCkEDIQ4gCiANjF8EQEEAIQ4FIApDAAAAAF8EQEEBIQ4FIAogDV8EQEECIQ4LCwsgDiAOQQF2cyEP" + "IAZBAnYhEEEDIAZBA3FrQQF0IREgBCAQaiAEIBBqLQAAIA8gEXRyOgAAIAZBAWohBgwACwsgBQuVAQMG" + "fwF+AX9B/wEhB0IAIQhBACEJQQAhAgJAA0AgAiABTw0BIAJBAnYhA0EDIAJBA3FrQQF0IQQgACADai0A" + "ACAEdkEDcSEFIAVBAnEgBUEBdiAFQQFxc3IhBiAGIAdHBEAgBiEHIAlBIEkEQCAIIAatQT4gCUEBdGut" + "hoQhCCAJQQFqIQkLCyACQQFqIQIMAAsLIAgL";
function b64ToBytes(b64) {
  const bin = atob(b64.replace(/\s+/g, ""));
  const out = new Uint8Array(bin.length);
  for (let i = 0;i < bin.length; i++)
    out[i] = bin.charCodeAt(i);
  return out;
}
var Q2_INPUT_OFFSET = 262144;
var Q2_OUTPUT_OFFSET = 65536;
var kernelPromise = null;
function getKernel() {
  kernelPromise ??= instantiate();
  return kernelPromise;
}
async function instantiate() {
  const bytes = b64ToBytes(WASM_B64);
  const { instance } = await WebAssembly.instantiate(bytes.buffer, {});
  const e = instance.exports;
  return {
    memory: e.mem,
    quantise(inputOffset, seqLen, n, dtype, outOffset) {
      return e.q2_quantise(inputOffset, seqLen, n, dtype, outOffset);
    },
    key(packedOffset, n) {
      return e.q2_key(packedOffset, n);
    }
  };
}
function q2EncodeDirect(vec, n) {
  const tau = 0.6745 / Math.sqrt(n);
  const nBytes = n >> 2;
  const packed = new Uint8Array(nBytes);
  for (let d = 0;d < n; d++) {
    const v = vec[d] ?? 0;
    let sym;
    if (v <= -tau)
      sym = 0;
    else if (v <= 0)
      sym = 1;
    else if (v <= tau)
      sym = 2;
    else
      sym = 3;
    const g = sym ^ sym >> 1;
    const byteIdx = d >> 2;
    const shift = 3 - (d & 3) << 1;
    packed[byteIdx] |= g << shift;
  }
  return { packed, key: q2KeyDirect(packed, n) };
}
function q2KeyDirect(packed, n) {
  let key = 0n;
  let trans = 0;
  let prev = 255;
  for (let d = 0;d < n; d++) {
    const byteIdx = d >> 2;
    const shift = 3 - (d & 3) << 1;
    const g = (packed[byteIdx] ?? 0) >> shift & 3;
    const z = g & 2 | g >> 1 ^ g & 1;
    if (z !== prev) {
      prev = z;
      if (trans < 32) {
        key |= BigInt(z) << BigInt(62 - 2 * trans);
        trans++;
      }
    }
  }
  return key;
}
function meanPoolAndNormalise(data, seqLen, n) {
  const v = new Float32Array(n);
  if (seqLen <= 0) {
    throw new Error(`meanPoolAndNormalise: seqLen must be > 0, got ${seqLen}`);
  }
  for (let s = 0;s < seqLen; s++) {
    for (let d = 0;d < n; d++) {
      v[d] += data[s * n + d] ?? 0;
    }
  }
  for (let d = 0;d < n; d++)
    v[d] /= seqLen;
  let normSq = 0;
  for (let d = 0;d < n; d++) {
    const x = v[d];
    normSq += x * x;
  }
  if (normSq > 0.0000000000000001) {
    const normInv = 1 / Math.sqrt(normSq);
    for (let d = 0;d < n; d++)
      v[d] *= normInv;
  }
  return v;
}

// src/opfs.ts
var OPFS_DIR = "q2";
var LOCALSTORAGE_KEY = "q2_opfs_file_map_v1";
function loadMapping() {
  try {
    const raw = localStorage.getItem(LOCALSTORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}
function saveMapping(mapping) {
  try {
    localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(mapping));
  } catch {}
}
async function digestHex(data) {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  const hashBuf = await crypto.subtle.digest("SHA-256", view);
  return Array.from(new Uint8Array(hashBuf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function getOpfsRoot() {
  const nav = navigator;
  if (nav.storage && typeof nav.storage.getDirectory === "function") {
    return await nav.storage.getDirectory();
  }
  const win = window;
  if (win.originPrivateFileSystem) {
    return win.originPrivateFileSystem;
  }
  return null;
}
async function ensureDir(pathSegments) {
  const root = await getOpfsRoot();
  if (!root)
    return null;
  let dir = root;
  for (const seg of pathSegments) {
    dir = await dir.getDirectoryHandle(seg, { create: true });
  }
  return dir;
}
async function writeOpfsFile(path, data) {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir)
    throw new Error("OPFS is not available in this environment");
  const name = path.replace(/^\/+|\/+$/g, "");
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  await writable.write(new Uint8Array(data));
  await writable.close();
}
async function readOpfsFile(path) {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir)
    throw new Error("OPFS is not available in this environment");
  const name = path.replace(/^\/+|\/+$/g, "");
  const handle = await dir.getFileHandle(name, { create: false });
  const file = await handle.getFile();
  return new Uint8Array(await file.arrayBuffer());
}
function hasRemoveEntry(dir) {
  return typeof dir.removeEntry === "function";
}
async function deleteOpfsFile(path) {
  const dir = await ensureDir([OPFS_DIR]);
  if (!dir)
    throw new Error("OPFS is not available in this environment");
  const name = path.replace(/^\/+|\/+$/g, "");
  if (hasRemoveEntry(dir)) {
    await dir.removeEntry(name, { recursive: false });
  } else {
    const legacyDir = dir;
    if (typeof legacyDir.remove === "function") {
      await legacyDir.remove(name);
    }
  }
}
function ensureName(name, fallback) {
  if (name && name.trim())
    return name.trim();
  return fallback;
}
async function storeFile(file, name, url) {
  const buffer = await file.arrayBuffer();
  const hash = await digestHex(buffer);
  const primaryName = ensureName(name, file.name ?? hash);
  const mapping = loadMapping();
  const now = Date.now();
  const meta = {
    hash,
    name: primaryName,
    size: buffer.byteLength,
    created: now,
    ...url !== undefined && { url }
  };
  mapping[hash] = meta;
  saveMapping(mapping);
  try {
    await writeOpfsFile(hash, buffer);
  } catch {}
  return meta;
}
async function storeFromUrl(url, name) {
  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const hash = await digestHex(buffer);
  const primaryName = ensureName(name, new URL(url).pathname.split("/").pop() || hash);
  const mapping = loadMapping();
  const now = Date.now();
  const meta = { hash, name: primaryName, size: buffer.byteLength, created: now, url };
  mapping[hash] = meta;
  saveMapping(mapping);
  try {
    await writeOpfsFile(hash, buffer);
  } catch {}
  return meta;
}
function listStoredFiles() {
  return Object.values(loadMapping()).sort((a, b) => b.created - a.created);
}
async function getStoredFile(hash) {
  try {
    const data = await readOpfsFile(hash);
    return data;
  } catch {
    return null;
  }
}
async function deleteStoredFile(hash) {
  const mapping = loadMapping();
  delete mapping[hash];
  saveMapping(mapping);
  try {
    await deleteOpfsFile(hash);
  } catch {}
}
function isOpfsAvailable() {
  const nav = navigator;
  if (nav.storage && typeof nav.storage.getDirectory === "function")
    return true;
  const win = window;
  return Boolean(win.originPrivateFileSystem);
}

// src/settings.ts
var DEFAULT_SETTINGS = {
  apiToken: "",
  dtype: "q4",
  filterLibrary: "transformers.js"
};
var SETTINGS_KEY = "q2_settings";
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw)
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {}
  return { ...DEFAULT_SETTINGS };
}
function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {}
}

// src/hf.ts
function formatCount(n) {
  if (n >= 1e6)
    return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1000)
    return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}
async function fetchHFModels(query, settings) {
  const params = new URLSearchParams({
    pipeline_tag: "text-generation",
    sort: "downloads",
    direction: "-1",
    limit: "20"
  });
  if (query.trim())
    params.set("search", query.trim());
  if (settings.filterLibrary)
    params.set("library", settings.filterLibrary);
  const headers = { Accept: "application/json" };
  if (settings.apiToken)
    headers["Authorization"] = `Bearer ${settings.apiToken}`;
  const res = await fetch(`https://huggingface.co/api/models?${params}`, { headers });
  if (!res.ok)
    throw new Error(`HF API ${res.status}: ${res.statusText}`);
  return await res.json();
}

// src/chat-render.ts
function splitThinkBlocks(raw) {
  const parts = [];
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  let lastIndex = 0;
  let m;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > lastIndex) {
      parts.push({ type: "text", text: raw.slice(lastIndex, m.index) });
    }
    parts.push({ type: "think", text: m[1] ?? "" });
    lastIndex = re.lastIndex;
  }
  if (lastIndex < raw.length) {
    const tail = raw.slice(lastIndex);
    const openTag = tail.indexOf("<think>");
    if (openTag !== -1) {
      if (openTag > 0)
        parts.push({ type: "text", text: tail.slice(0, openTag) });
      parts.push({ type: "think", text: tail.slice(openTag + 7) });
    } else {
      parts.push({ type: "text", text: tail });
    }
  }
  return parts;
}
function stripThinkTags(raw) {
  return raw.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*$/g, "").trim();
}
function escapeAndFormatText(text) {
  let s = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const codeBlocks = [];
  s = s.replace(/```([\s\S]*?)```/g, (_match, codeContent) => {
    const index = codeBlocks.length;
    codeBlocks.push(`<pre class="code-block"><code>${codeContent}</code></pre>`);
    return `__CODE_BLOCK_${index}__`;
  });
  s = s.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*([^*]|$)/g, "$1<em>$2</em>$3");
  s = s.replace(/\n/g, "<br>");
  s = s.replace(/__CODE_BLOCK_(\d+)__/g, (_match, index) => {
    const i = Number(index);
    return codeBlocks[i] ?? "";
  });
  return s;
}

// src/embed-panel.ts
function min(arr) {
  let m = Infinity;
  for (const v of arr)
    if (v < m)
      m = v;
  return m;
}
function max(arr) {
  let m = -Infinity;
  for (const v of arr)
    if (v > m)
      m = v;
  return m;
}
function renderEmbeddingHeatmap(data, seqLen, hiddenDim, canvas) {
  const W = Math.min(seqLen, canvas.clientWidth || 320);
  const H = 64;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx)
    return;
  const rowsPerCell = Math.ceil(hiddenDim / H);
  const colsPerCell = Math.ceil(seqLen / W);
  const minVal = min(data);
  const maxVal = max(data);
  const range = maxVal - minVal || 1;
  for (let row = 0;row < H; row++) {
    for (let col = 0;col < W; col++) {
      let sum = 0;
      let count = 0;
      for (let d = row * rowsPerCell;d < Math.min((row + 1) * rowsPerCell, hiddenDim); d++) {
        for (let s = col * colsPerCell;s < Math.min((col + 1) * colsPerCell, seqLen); s++) {
          sum += data[s * hiddenDim + d];
          count++;
        }
      }
      const v = count ? sum / count : 0;
      const t = (v - minVal) / range;
      const r = t > 0.5 ? 255 : Math.round(t * 2 * 255);
      const g = t > 0.5 ? Math.round((1 - t) * 2 * 255) : Math.round(t * 2 * 255);
      const b = t < 0.5 ? 255 : Math.round((1 - t) * 2 * 255);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(col, row, 1, 1);
    }
  }
}
function renderQ2Result(packed, key, n, statsEl) {
  const hexBytes = Array.from(packed.slice(0, 8)).map((b) => b.toString(16).padStart(2, "0")).join("");
  const ellipsis = packed.length > 8 ? "…" : "";
  const keyHex = key.toString(16).padStart(16, "0");
  statsEl.textContent += `
Q²: [${hexBytes}${ellipsis}] (${n >> 2} bytes, ${n} dims)  key=0x${keyHex}`;
}

// src/app.ts
var SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant. " + "Think carefully before answering.";
var DEFAULT_CONFIG = {
  max_new_tokens: 2048,
  temperature: 0,
  repetition_penalty: 1.1
};
function $(selector) {
  const el = document.querySelector(selector);
  if (!el)
    throw new Error(`Element not found: ${selector}`);
  return el;
}
var loadScreen = $("#load-screen");
var loadStatus = $("#load-status");
var loadBar = $("#load-bar-fill");
var chatApp = $("#chat-app");
var messagesEl = $("#messages");
var inputEl = $("#user-input");
var sendBtn = $("#send-btn");
var stopBtn = $("#stop-btn");
var embeddingPanel = $("#embedding-panel");
var embeddingCanvas = $("#embedding-canvas");
var embeddingStats = $("#embedding-stats");
var localFileDrop = $("#local-file-drop");
var localFileUrl = $("#local-file-url");
var localFileAddBtn = $("#local-file-add");
var localFilesList = $("#local-files-list");
var localFileDropDefaultText = localFileDrop.textContent ?? "";
var maxTokensEl = $("#max-tokens");
var temperatureEl = $("#temperature");
var tempValueEl = $("#temp-value");
var repPenaltyEl = $("#rep-penalty");
var repValueEl = $("#rep-value");
var modelPickerEl = $("#model-picker");
var loadProgressEl = $("#load-progress");
var modelSearchEl = $("#model-search");
var modelListEl = $("#model-list");
var modelCustomIdEl = $("#model-custom-id");
var loadBtnEl = $("#load-btn");
var headerTitleEl = $("#header-title");
var sidebarModelTagEl = $("#sidebar-model-tag");
var worker = null;
var modelReady = false;
var isGenerating = false;
var currentSettings = loadSettings();
var selectedModelId = "";
var history = [
  { role: "system", content: SYSTEM_PROMPT }
];
var activeBubble = null;
var activeRawText = "";
var searchTimer = null;
async function refreshModelList(query, autoSelectFirst = false) {
  modelListEl.innerHTML = '<li class="model-list-status">Searching models…</li>';
  try {
    const models = await fetchHFModels(query, currentSettings);
    if (models.length === 0) {
      modelListEl.innerHTML = '<li class="model-list-status">No models found. Try a different search.</li>';
      return;
    }
    modelListEl.innerHTML = "";
    for (const model of models) {
      renderModelItem(model);
    }
    if (autoSelectFirst && !selectedModelId && models[0]) {
      selectModel(models[0].id);
    }
  } catch (err) {
    const li = document.createElement("li");
    li.className = "model-list-status model-list-error";
    li.textContent = String(err);
    const retryBtn = document.createElement("button");
    retryBtn.className = "model-list-retry";
    retryBtn.textContent = "Retry";
    retryBtn.addEventListener("click", () => void refreshModelList(query, autoSelectFirst));
    li.appendChild(retryBtn);
    modelListEl.innerHTML = "";
    modelListEl.appendChild(li);
  }
}
function renderModelItem(model) {
  const li = document.createElement("li");
  li.role = "option";
  li.className = "model-item" + (model.id === selectedModelId ? " selected" : "");
  li.setAttribute("aria-selected", model.id === selectedModelId ? "true" : "false");
  li.dataset["modelId"] = model.id;
  const slashIdx = model.id.indexOf("/");
  const author = slashIdx !== -1 ? model.id.slice(0, slashIdx) : "";
  const name = slashIdx !== -1 ? model.id.slice(slashIdx + 1) : model.id;
  const info = document.createElement("div");
  info.className = "model-item-info";
  const labelEl = document.createElement("span");
  labelEl.className = "model-item-label";
  labelEl.textContent = name;
  const authorEl = document.createElement("span");
  authorEl.className = "model-item-author";
  authorEl.textContent = author;
  info.appendChild(labelEl);
  info.appendChild(authorEl);
  const stats = document.createElement("div");
  stats.className = "model-item-stats";
  const dlEl = document.createElement("span");
  dlEl.className = "model-item-stat";
  dlEl.title = "Downloads";
  dlEl.textContent = `↓ ${formatCount(model.downloads)}`;
  const likeEl = document.createElement("span");
  likeEl.className = "model-item-stat";
  likeEl.title = "Likes";
  likeEl.textContent = `♥ ${formatCount(model.likes)}`;
  stats.appendChild(dlEl);
  stats.appendChild(likeEl);
  li.appendChild(info);
  li.appendChild(stats);
  li.addEventListener("click", () => selectModel(model.id));
  modelListEl.appendChild(li);
}
function selectModel(modelId) {
  selectedModelId = modelId;
  modelCustomIdEl.value = "";
  document.querySelectorAll(".model-item").forEach((item) => {
    const selected = item.dataset["modelId"] === modelId;
    item.classList.toggle("selected", selected);
    item.setAttribute("aria-selected", selected ? "true" : "false");
  });
  loadBtnEl.disabled = false;
}
function initModelPicker() {
  refreshModelList("", true);
  modelSearchEl.addEventListener("input", () => {
    const q = modelSearchEl.value;
    if (searchTimer)
      clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      refreshModelList(q);
      searchTimer = null;
    }, 400);
  });
  modelCustomIdEl.addEventListener("input", () => {
    const val = modelCustomIdEl.value.trim();
    if (val) {
      document.querySelectorAll(".model-item").forEach((item) => {
        item.classList.remove("selected");
        item.setAttribute("aria-selected", "false");
      });
      loadBtnEl.disabled = false;
    } else {
      loadBtnEl.disabled = !selectedModelId;
    }
  });
  modelCustomIdEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      triggerLoad();
    }
  });
  loadBtnEl.addEventListener("click", triggerLoad);
  initSettingsPanel();
  initLocalFileStore();
}
function initSettingsPanel() {
  const toggleBtn = $("#settings-toggle");
  const panel = $("#settings-panel");
  const tokenEl = $("#hf-token");
  const dtypeEl = $("#model-dtype");
  const libraryEl = $("#filter-library");
  tokenEl.value = currentSettings.apiToken;
  dtypeEl.value = currentSettings.dtype;
  libraryEl.value = currentSettings.filterLibrary;
  toggleBtn.addEventListener("click", () => {
    const isHidden = panel.classList.toggle("hidden");
    toggleBtn.setAttribute("aria-expanded", String(!isHidden));
  });
  tokenEl.addEventListener("change", () => {
    currentSettings.apiToken = tokenEl.value.trim();
    saveSettings(currentSettings);
  });
  dtypeEl.addEventListener("change", () => {
    currentSettings.dtype = dtypeEl.value;
    saveSettings(currentSettings);
  });
  libraryEl.addEventListener("change", () => {
    currentSettings.filterLibrary = libraryEl.value;
    saveSettings(currentSettings);
    refreshModelList(modelSearchEl.value);
  });
}
function setLocalFileStatus(text, durationMs = 2500) {
  localFileDrop.textContent = text;
  setTimeout(() => {
    localFileDrop.textContent = localFileDropDefaultText;
  }, durationMs);
}
function renderLocalFileList() {
  const files = listStoredFiles();
  localFilesList.innerHTML = "";
  if (files.length === 0) {
    const li = document.createElement("li");
    li.className = "local-file-item";
    li.textContent = "No local files stored yet.";
    localFilesList.appendChild(li);
    return;
  }
  for (const file of files) {
    const li = document.createElement("li");
    li.className = "local-file-item";
    const nameSpan = document.createElement("span");
    nameSpan.className = "local-file-name";
    nameSpan.textContent = file.name;
    const actions = document.createElement("span");
    actions.className = "local-file-actions";
    const downloadBtn = document.createElement("button");
    downloadBtn.type = "button";
    downloadBtn.textContent = "Download";
    downloadBtn.addEventListener("click", async () => {
      const data = await getStoredFile(file.hash);
      if (!data) {
        setLocalFileStatus("File not available in OPFS.", 3000);
        return;
      }
      const blob = new Blob([data]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name || file.hash;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    });
    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", async () => {
      await deleteStoredFile(file.hash);
      renderLocalFileList();
      setLocalFileStatus("Removed from local storage.", 2000);
    });
    actions.appendChild(downloadBtn);
    actions.appendChild(deleteBtn);
    li.appendChild(nameSpan);
    li.appendChild(actions);
    localFilesList.appendChild(li);
  }
}
async function handleLocalFile(file) {
  try {
    const meta = await storeFile(file, file.name);
    renderLocalFileList();
    setLocalFileStatus(`Saved ${meta.name}`);
  } catch (err) {
    setLocalFileStatus(`Error saving file: ${String(err)}`);
  }
}
async function handleLocalUrl(rawUrl) {
  const url = rawUrl.trim();
  if (!url)
    return;
  try {
    const meta = await storeFromUrl(url);
    renderLocalFileList();
    setLocalFileStatus(`Fetched and saved ${meta.name}`);
  } catch (err) {
    setLocalFileStatus(`Error fetching URL: ${String(err)}`);
  }
}
function initLocalFileStore() {
  if (!isOpfsAvailable()) {
    setLocalFileStatus("OPFS not supported in this browser.");
  }
  renderLocalFileList();
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", async () => {
    if (fileInput.files?.length) {
      await handleLocalFile(fileInput.files[0]);
    }
  });
  document.body.appendChild(fileInput);
  localFileDrop.addEventListener("click", () => fileInput.click());
  localFileDrop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  localFileDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
    localFileDrop.classList.add("dragover");
  });
  localFileDrop.addEventListener("dragleave", () => {
    localFileDrop.classList.remove("dragover");
  });
  localFileDrop.addEventListener("drop", async (e) => {
    e.preventDefault();
    localFileDrop.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) {
      await handleLocalFile(e.dataTransfer.files[0]);
    }
  });
  localFileAddBtn.addEventListener("click", async () => {
    await handleLocalUrl(localFileUrl.value);
  });
}
function triggerLoad() {
  const customId = modelCustomIdEl.value.trim();
  const modelId = customId || selectedModelId;
  if (!modelId)
    return;
  startWithModel(modelId);
}
function startWithModel(modelId) {
  selectedModelId = modelId;
  modelPickerEl.classList.add("hidden");
  loadProgressEl.classList.remove("hidden");
  initWorker(modelId);
}
function initWorker(modelId) {
  const workerUrl = globalThis.__Q2_WORKER_URL__ ?? new URL("./worker.js", import.meta.url).toString();
  worker = new Worker(workerUrl, {
    type: "module"
  });
  worker.addEventListener("message", (e) => {
    handleWorkerMessage(e.data);
  });
  worker.addEventListener("error", (e) => {
    showError(`Worker error: ${e.message}`);
  });
  const loadMsg = currentSettings.apiToken ? { type: "load", modelId, dtype: currentSettings.dtype, apiToken: currentSettings.apiToken } : { type: "load", modelId, dtype: currentSettings.dtype };
  postToWorker(loadMsg);
}
function postToWorker(msg) {
  worker?.postMessage(msg);
}
function handleWorkerMessage(msg) {
  switch (msg.type) {
    case "status":
      onStatus(msg.status, msg.detail);
      break;
    case "progress":
      onProgress(msg.file, msg.loaded, msg.total);
      break;
    case "token":
      onToken(msg.token);
      break;
    case "embedding":
      onEmbedding(msg);
      break;
    case "done":
      onDone();
      break;
    case "error":
      if (!modelReady) {
        loadStatus.textContent = `Error loading model: ${msg.message}`;
        loadScreen.classList.remove("hidden");
        chatApp.classList.add("hidden");
      } else {
        showError(msg.message);
      }
      onDone();
      break;
  }
}
function onStatus(status, detail) {
  if (status === "ready") {
    modelReady = true;
    loadScreen.classList.add("hidden");
    chatApp.classList.remove("hidden");
    const displayName = selectedModelId.split("/").at(-1) ?? selectedModelId;
    headerTitleEl.textContent = `${displayName} · ${currentSettings.dtype.toUpperCase()} ONNX`;
    sidebarModelTagEl.textContent = displayName;
    inputEl.focus();
  } else if (status === "loading") {
    loadStatus.textContent = detail ?? "Loading model…";
  }
}
function onProgress(file, loaded, total) {
  if (total > 0) {
    const pct = Math.round(loaded / total * 100);
    loadBar.style.width = `${pct}%`;
    const statusText = file ? `Downloading ${file.split("/").pop() ?? file} — ${pct}%` : `Downloading… ${pct}%`;
    loadStatus.textContent = statusText;
    loadBar.setAttribute("aria-valuenow", String(pct));
    loadBar.setAttribute("aria-valuetext", statusText);
  } else {
    const statusText = file ? `Loading ${file.split("/").pop() ?? file}…` : "Loading…";
    loadStatus.textContent = statusText;
    loadBar.removeAttribute("aria-valuenow");
    loadBar.setAttribute("aria-valuetext", statusText);
  }
}
var tokenRenderScheduled = false;
function scheduleBubbleRender() {
  if (!activeBubble) {
    tokenRenderScheduled = false;
    return;
  }
  if (tokenRenderScheduled)
    return;
  tokenRenderScheduled = true;
  requestAnimationFrame(() => {
    if (!activeBubble) {
      tokenRenderScheduled = false;
      return;
    }
    renderBubble(activeBubble, activeRawText);
    scrollToBottom();
    tokenRenderScheduled = false;
  });
}
function onToken(token) {
  if (!activeBubble)
    return;
  activeRawText += token;
  scheduleBubbleRender();
}
function onEmbedding(msg) {
  embeddingPanel.classList.remove("hidden");
  const { seqLen, hiddenDim, dtype } = msg;
  const expectedElements = seqLen * hiddenDim;
  let floats = null;
  if (dtype === "fp32") {
    if (msg.data.byteLength % 4 !== 0) {
      console.warn(`Embedding fp32 data has byteLength=${msg.data.byteLength}, which is not a multiple of 4; skipping Float32 view.`);
    } else {
      const view = new Float32Array(msg.data);
      if (view.length !== expectedElements) {
        console.warn(`Embedding fp32 data has length=${view.length}, expected=${expectedElements} (seqLen=${seqLen}, hiddenDim=${hiddenDim}); skipping Float32 view.`);
      } else {
        floats = view;
      }
    }
  }
  if (floats) {
    renderEmbeddingHeatmap2(floats, seqLen, hiddenDim);
    embeddingStats.textContent = `Shape: [${seqLen} × ${hiddenDim}]  dtype=${dtype}  ` + `min=${min(floats).toFixed(3)}  max=${max(floats).toFixed(3)}`;
  } else {
    embeddingStats.textContent = `Shape: [${seqLen} × ${hiddenDim}]  dtype=${dtype}  stats=unavailable`;
  }
  const n = hiddenDim;
  const dtypeId = DTYPE_TO_Q2[dtype] ?? Q2_DTYPE_FP32;
  (async () => {
    try {
      const kernel = await getKernel();
      const mem = new Uint8Array(kernel.memory.buffer);
      const inputBytes = new Uint8Array(msg.data);
      mem.set(inputBytes, Q2_INPUT_OFFSET);
      kernel.quantise(Q2_INPUT_OFFSET, seqLen, n, dtypeId, Q2_OUTPUT_OFFSET);
      const rawKey = kernel.key(Q2_OUTPUT_OFFSET, n);
      const key = BigInt.asUintN(64, rawKey);
      const packed = new Uint8Array(kernel.memory.buffer, Q2_OUTPUT_OFFSET, n >> 2);
      renderQ2Result2(packed, key, n);
    } catch {
      if (dtype !== "fp32") {
        console.warn(`Q² TS fallback: dtype=${dtype} requires WASM kernel; skipping.`);
        return;
      }
      const vec = meanPoolAndNormalise(new Float32Array(msg.data), seqLen, n);
      const { packed, key } = q2EncodeDirect(vec, n);
      renderQ2Result2(packed, BigInt.asUintN(64, key), n);
    }
  })();
}
function onDone() {
  isGenerating = false;
  sendBtn.disabled = false;
  sendBtn.classList.remove("hidden");
  stopBtn.classList.add("hidden");
  if (activeBubble) {
    history.push({ role: "assistant", content: stripThinkTags(activeRawText) });
    renderBubble(activeBubble, activeRawText);
    activeBubble = null;
    activeRawText = "";
  }
  inputEl.disabled = false;
  inputEl.focus();
}
function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !modelReady || isGenerating)
    return;
  history.push({ role: "user", content: text });
  appendUserBubble(text);
  inputEl.value = "";
  autoResizeTextarea();
  activeBubble = appendAssistantBubble();
  activeRawText = "";
  isGenerating = true;
  sendBtn.disabled = true;
  sendBtn.classList.add("hidden");
  stopBtn.classList.remove("hidden");
  inputEl.disabled = true;
  postToWorker({
    type: "generate",
    messages: history.slice(),
    config: readConfig()
  });
}
function stopGeneration() {
  postToWorker({ type: "abort" });
}
function readConfig() {
  return {
    max_new_tokens: parseInt(maxTokensEl.value, 10) || DEFAULT_CONFIG.max_new_tokens,
    temperature: parseFloat(temperatureEl.value) || DEFAULT_CONFIG.temperature,
    repetition_penalty: parseFloat(repPenaltyEl.value) || DEFAULT_CONFIG.repetition_penalty
  };
}
function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "message-row user";
  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}
function appendAssistantBubble() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  bubble.innerHTML = '<span class="cursor"></span>';
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return bubble;
}
function renderBubble(bubble, raw) {
  const parts = splitThinkBlocks(raw);
  bubble.innerHTML = "";
  for (const part of parts) {
    if (part.type === "think") {
      const details = document.createElement("details");
      details.className = "think-block";
      const summary = document.createElement("summary");
      summary.textContent = "\uD83D\uDCAD Thinking…";
      const pre = document.createElement("pre");
      pre.className = "think-content";
      pre.textContent = part.text;
      details.appendChild(summary);
      details.appendChild(pre);
      bubble.appendChild(details);
    } else {
      const div = document.createElement("div");
      div.innerHTML = escapeAndFormatText(part.text);
      bubble.appendChild(div);
    }
  }
  if (isGenerating) {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    bubble.appendChild(cursor);
  }
}
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function autoResizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
}
function showError(message) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble error";
  bubble.textContent = `⚠️ ${message}`;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}
function renderEmbeddingHeatmap2(data, seqLen, hiddenDim) {
  renderEmbeddingHeatmap(data, seqLen, hiddenDim, embeddingCanvas);
}
function renderQ2Result2(packed, key, n) {
  renderQ2Result(packed, key, n, embeddingStats);
}
sendBtn.addEventListener("click", sendMessage);
stopBtn.addEventListener("click", stopGeneration);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
inputEl.addEventListener("input", autoResizeTextarea);
temperatureEl.addEventListener("input", () => {
  tempValueEl.textContent = parseFloat(temperatureEl.value).toFixed(2);
});
repPenaltyEl.addEventListener("input", () => {
  repValueEl.textContent = parseFloat(repPenaltyEl.value).toFixed(2);
});
if (!globalThis.__Q2_SKIP_AUTO_INIT__) {
  initModelPicker();
}
export {
  worker,
  stripThinkTags,
  stopGeneration,
  startWithModel,
  splitThinkBlocks,
  sendMessage,
  selectedModelId,
  selectModel,
  saveSettings,
  renderQ2Result2 as renderQ2Result,
  renderEmbeddingHeatmap2 as renderEmbeddingHeatmap,
  renderBubble,
  readConfig,
  onToken,
  onStatus,
  onProgress,
  onEmbedding,
  onDone,
  min,
  max,
  loadSettings,
  initWorker,
  initModelPicker,
  initLocalFileStore,
  handleWorkerMessage,
  handleLocalUrl,
  formatCount,
  fetchHFModels,
  escapeAndFormatText,
  appendAssistantBubble
};

//# debugId=6F094BD97B9D8F2664756E2164756E21
//# sourceMappingURL=app.js.map
