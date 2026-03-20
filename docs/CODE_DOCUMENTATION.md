# q2 Project Code Documentation

This document provides a comprehensive, module-by-module code reference for the `q2` project. It is derived from a line-level review of the source code and includes function behavior, contracts, and architectural notes.

---

## 1. Project Overview

`q2` is a browser-native quantisation/retrieval/embedding research tool for transformer models. It uses:
- **transformers.js** ONNX backend, with WebNN/WebGPU/WebGL/WASM fallbacks
- **Q² quantisation**: quaternary quantisation of hidden states to Z₄ values (A/B/C/D or CGAT)
- **Embed-to-key**: 64-bit transition key from run-reduced Z₄ transition traces
- Benchmarks of predictions P1..P14 via synthetic sequence tests

Core workflow:
1. load model in worker
2. generate text + hidden states from layer 9
3. run Q² quantisation (WASM kernel or TS fallback)
4. publish embedding heatmap + q2 key + panel

---

## 2. Source modules

### 2.1 `src/types.ts`
Shared types and worker message shapes.
- `Dtype`: `'q4' | 'q8' | 'fp16' | 'fp32'`
- `FilterLibrary`: `'transformers.js' | 'onnx' | ''`
- `Q2KeyDisplayMode`: `'q2' | 'cgAt' | 'hex'`
- `LoadModelMsg`, `GenerateMsg`, `AbortMsg` for `WorkerInMsg`
- `StatusMsg`, `ProgressMsg`, `TokenMsg`, `EmbeddingMsg`, `DoneMsg`, `ErrorMsg` for `WorkerOutMsg`
- `ChatMessage`: role/content
- `GenerationConfig`: `max_new_tokens`, `temperature`, `repetition_penalty`

### 2.2 `src/settings.ts`
Persistence of app settings in localStorage.
- `AppSettings` fields:
  - `apiToken`, `dtype`, `filterLibrary`, `q2KeyDisplayMode`
  - `defaultChatModel`, `benchModelT2`, `benchModelT3`, `benchModelT4`
- `DEFAULT_SETTINGS` with defaults
- `loadSettings()`: parse existing JSON; fallback defaults; errors ignored safe
- `saveSettings(settings)`: JSON stringify + localStorage

### 2.3 `src/chat-render.ts`
UI text formatting for streaming assistant chat.
- `Part` type union: `text|think`
- `splitThinkBlocks(raw)`: parse `<think>...</think>` and partial open tags into parts.
- `stripThinkTags(raw)`: removes full and partial `<think>` traces (for history) and trims.
- `escapeAndFormatText(text)`: HTTP-escape + transform
  - fenced code blocks ```...```
  - `inline code`
  - `**bold**`, `*italic*`
  - newlines → `<br>`

### 2.4 `src/embed-panel.ts`
Rendering helpers for embedding panel.
- `min(arr)`, `max(arr)` for `Float32Array`
- `Q2KeyDisplayMode` alias
- `formatQ2KeyDisplay(packed, n, key, mode)`
  - `q2` mode: `A/B/C/D` sequence from run-reduced transition keys
  - `cgAt` mode: `G/A/C/T`
  - `hex` mode: raw 64-bit key
- `renderEmbeddingHeatmap(data, seqLen, hiddenDim, canvas)`
  - draws `H=64` rows, `W=min(seqLen, canvasWidth)` columns
  - Bin average from `seqLen × hiddenDim` data
  - colour map blue-white-red
- `renderQ2Result(packed, key, n, statsEl, mode)`
  - append 1st 8 bytes hex summary and key display

### 2.5 `src/hf.ts`
HuggingFace Hub utilities.
- `HFModel`: `id`, `downloads`, `likes`, `tags`.
- `formatCount(n)`: compact formatting (`1.2K`, `3.4M`)
- `parseModelParameterCount(modelId)`: parse suffix patterns (`k,K,m,M,b,B`)
- `isModelWithinParameterLimit(model)`: `<= 2e9` or unknown.
- `fetchHFModels(query, settings)`: HF `/api/models` with filters, token support, parameter limit.

### 2.6 `src/q2.ts`
Q² kernel wrapper and TS fallback.

#### Constants
- `Q2Dtype` enum `0..4` (fp32, fp16, q8, q4, q2)
- `DTYPE_TO_Q2` mapping from dtype string
- `Q2_INPUT_OFFSET = 0x40000`, `Q2_OUTPUT_OFFSET = 0x10000`
- `WASM_B64`: embedded Q² WASM binary from `src/q2.wat`

#### Runtime utilities
- `b64ToBytes(b64)`: base64→Uint8Array

#### Kernel interface
- `Q2Kernel` type: `memory`, `quantise(...)`, `key(...)`
- `kernelPromise: Promise<Q2Kernel> | null`
- `getKernel()`: singleton promise
- `instantiate()`: WebAssembly.instantiate bytes, returns object with `quantise`, `key`

#### TS fallback results
- `Q2Result`: `{ packed: Uint8Array; key: bigint }`

#### q2 high-level operations
- `q2EncodeDirect(vec, n)`:
  - threshold `tau=0.6745/sqrt(n)`
  - for each `v`: sym
    - v <= -tau → 0 (A)
    - v <= 0 → 1 (B)
    - v <= tau → 2 (C)
    - else → 3 (D)
  - Gray encode `g = sym ^ (sym >> 1)`
  - pack 4 two-bit symbols per byte MSB-first
  - compute `key` via `q2KeyDirect` and return
- `q2KeyDirect(packed, n)`:
  - decode source Gray bytes to Z₄ with `z = (g & 2) | ((g >> 1) ^ (g & 1))`
  - run-reduce transitions and encode first ≤32 changes into 64-bit key MSB-aligned
- `l2Normalise(data, n)`:
  - length-normalise first `n` elements; pad missing elements with 0; ignore extra.

### 2.7 `src/q2stats.ts`
Statistical primitives over Z₄ transition sequences.

#### Fundamental algebra
- `complement(z)`: Z₄ complement (flip low-order bit): `(z ^ 2) & 3`
- `leeDistance(a, b)` through circle metric
- `leeDistancePacked(a, b)` by popcount on XOR of packed q2 bytes
- `grayEncode(z)` / `grayDecode(g)` (2-bit Gray code and inverse)
- `unpackSymbols(packed, n)`: unpack packed bytes to full Z₄ sequence
- `runReduce(symbols)`: collapse consecutive duplicates

#### core predictive statistics
- `hairpinDensity(seq)`: ratio of i where r[i+1]==θ(r[i]) and r[i+2]==r[i]
- `complementBigramFreq(seq)`: fraction of i where r[i+1]==θ(r[i])
- `tripletFreqs(seq)` and `bigramFreqs(seq)` count n-grams in run-reduced seq
- `reverseComplementSeq(seq)` = reverse + complement
- `collisionStats(keys)` returns collisions/groups/rate
- `nullCollisionExpectation(n,bits=64)` approximate expected collisions

#### mutation classes
- `bigramType(a,b)` -> `'same' | 'Ti' | 'Tv1' | 'Tv2'`
- `weightedLeeDistanceSeq(a,b,weights)`: weighted sum by bigram type

#### structure scoring
- `nussinovScore(seq)` dynamic programming for non-crossing complement pairs (O(n^3), n<=2000)
- `leeDistanceSeq(a,b)` reduce across prefix

### 2.8 `src/worker.ts`
Web Worker with model load + generation + embeddings.

#### view environment / constants
- `isIOS()`: iOS / iPadOS detection, WebNN/WebGPU skip
- `workerLog(level,msg,args)` common logger
- `DEVICE_PRIORITY`: `['webgl','wasm']` for iOS; else `['webnn','webgpu','webgl','wasm']`
- `BACKEND_HANG_TIMEOUT_MS=30000`

#### state
- `pipe: TextGenerationPipeline | null`
- `stoppingCriteria` for clean abort
- `activeDtype` model activations dtype (returns `'fp32'` currently)

#### messaging
- `send(msg, transfer)`: post message to main thread

#### loader
- `toEmbeddingDtype(dtype)` currently forced `'fp32'` (transformers returns fp32 hidden states)
- `loadModel(modelId,dtype,apiToken)`
  - sets `env.accessToken` if provided
  - `send({status:'loading'})`
  - try each backend in `DEVICE_PRIORITY` with progress callback
  - hang guard interval fallback after 30s idle
  - on success wraps pipeline, sets `activeDtype`, `send({status:'ready'})`
  - on failure after all backends: `send({type:'error'})`

#### generate
- `generateResponse(messages, config)`
  - requires ready `pipe`
  - stream tokens via `TextStreamer(callback)` from pipeline
  - optional embedding output when `config.return_embeddings===true`
  - on done: extract hidden states layer 9, send `EmbeddingMsg` with packed `ArrayBuffer`
  - always sends `DoneMsg`; on error sends `ErrorMsg` unless interrupted

#### router
- handles `message` events:
  - `load` → loadModel
  - `generate` → generateResponse
  - `abort` → stoppingCriteria.interrupt()

### 2.9 `src/app.ts`
Main frontend glue: UI, model picker, worker orchestration, Q² and benchmarks.

#### exports
- Wiht re-exported helpers from submodules: `loadSettings, saveSettings, HFModel, fetchHFModels, formatCount, splitThinkBlocks, stripThinkTags, escapeAndFormatText, min, max`
- actions: `selectModel, initModelPicker, handleLocalUrl, startWithModel, initWorker, handleWorkerMessage, onStatus, onProgress, onToken, onEmbedding, onDone, sendMessage, stopGeneration, readConfig, appendAssistantBubble, renderBubble, renderEmbeddingHeatmap, renderQ2Result, switchTab, runBenchmarks`

#### DOM ref and state
- many selected DOM elements: panels, model controls, chat controls, local file store
- `history` chat message list; `activeBubble`, `activeRawText`
- `modelReady`, `isGenerating`, `selectedModelId`, etc.

#### model picker
- `refreshModelList(query, autoSelectFirst)` fetches HF models, render list, selects first
- `renderModelItem(model)` builds item and click handler
- `selectModel(modelId)`, `initModelPicker()` binds input listeners + settings + local files

#### settings panel
- `initSettingsPanel()` binds settings fields and writes `currentSettings` + save

#### local files (OPFS)
- `renderLocalFileList()`, `handleLocalFile(file)`, `handleLocalUrl(rawUrl)`
- `initLocalFileStore()` sets up drag/drop/click, file transfer

#### worker boot + message pipeline
- `triggerLoad()`, `startWithModel(modelId)` show overlay
- `initWorker(modelId)` create `Worker`, attach listeners
- `postToWorker(msg)` sends message with safe-log-sanitized payload
- `handleWorkerMessage` routes statuses/progress/token/embedding/done/error

#### status handlers
- `onStatus(status, detail)` updates DOM badges
- `onProgress(file, loaded, total)` updates load progress bar
- `onToken`, `onEmbedding`, `onDone` handle streaming + q2 pipeline

##### Q² path in `onEmbedding`
- parse `msg` for `seqLen`, `hiddenDim`, dtype
- render embedding heatmap for fp32 only
- run Q² kernel:
  - `kernel = await getKernel()`
  - copy activation buffer to WASM memory offset
  - `kernel.quantise(...)`, `kernel.key(...)`
  - read packed bytes and call `renderQ2Result`
- fallback if wasm fails:
  - parse fp32 data `Float32Array` last token
  - `l2Normalise`, `q2EncodeDirect`, `q2KeyDirect`, `renderQ2Result`

#### chat & generation
- `sendMessage()` reads `inputEl`, updates history, sets UI state, and posts `generate` message
- `stopGeneration()`, `readConfig()`
- `appendUserBubble`, `appendAssistantBubble`
- `renderBubble(bubble, raw)` handles `<think>` blocks and markdown formatting

#### utilities
- scroll, textarea resizing, show error bubble
- `renderEmbeddingHeatmap` and `renderQ2Result` wrappers
- `switchTab()` for tab navigation

#### benchmark suite
- random RNG sets, synthetic seq generators `callAndReturnSeq`, `linearSeq`, `dialecticalSeq`, etc.
- stat tests P1..P14 across T0..T5
- `runBenchmarks(suiteFilter?)` appends results table and final statistics

#### event bindings
- buttons, keyboard, control inputs, benchmark buttons
- auto-start `initModelPicker()` unless `__Q2_SKIP_AUTO_INIT__`

---

## 3. Q² Algorithm Notes

`q2EncodeDirect` implements quantisation of a length-n L2-normalized embedding (n power of 2) to n/4 bytes:
- threshold τ = Φ⁻¹(3/4)/√n ≈ 0.6745/√n
- symbol mapping value ranges to {0,1,2,3}
- Gray-coded symbols packed MSB-first by groups of 4
- key generated from run-length transitions (max 32 transitions) in MSB-aligned 64-bit word

`q2KeyDirect` is deterministic and produces 64-bit key per DESIGN.md §2.2.

`l2Normalise` is stable with vector zero guard.

---

## 4. Testing & Validation

- `test/q2.test.ts`, `test/q2stats.test.ts`, etc., verify exact kernel behavior and statistics.
- For WASM-enabled environments, `q2.ts` uses `getKernel()` with WASM fallback to TS reference.

---

## 5. How to use

1. Open `index.html` in compatible browser.
2. In Settings, select a model and click Load.
3. Send a prompt in Chat.
4. Embedding panel shows hidden–state heatmap and Q² key.
5. Benchmark tab runs P0..P5 predictions checks.

---

## 6. Practical API surface for embedding research

From `src/app.ts` plus exports, the reusable API is:
- `loadSettings`, `saveSettings`
- `fetchHFModels`, `formatCount`
- text render helpers: `splitThinkBlocks`, `stripThinkTags`, `escapeAndFormatText`
- `q2` core: `getKernel`, `l2Normalise`, `q2EncodeDirect`, `q2KeyDirect`
- `q2stats`: all above statistical functions

This document is intended for developers wishing to inspect behavior and extend algebraic analyses with minimal context switching.
