# NEXT

## Current status

- ✅ Added an OPFS-backed local file store (`src/opfs.ts`) that saves files by SHA-256 hash and keeps a filename/metadata mapping in `localStorage`.
- ✅ Added a small UI panel in the sidebar for:
  - Drag-and-drop file import
  - URL import
  - Listing stored files with Download + Delete actions
- ✅ Added tests for the OPFS store (`test/opfs.test.ts`) and updated app tests where needed.
- ✅ Refactored settings persistence into `src/settings.ts` and kept the existing public API by re-exporting helpers from `src/app.ts`.
- ✅ All tests pass (`bun run test`).

## What to do next

### 1) Make OPFS files usable for the pipeline
- Add a mechanism to load model/corpus artifacts from OPFS into the worker (e.g. pass a `modelUrl` or `localHash` to the worker and use fetch / Cache API / OPFS to load weights).
- Ensure the worker can read large files efficiently (streaming, if needed) and handle hash-based lookups.

### 2) Integrate OPFS store with Q² testing pipeline
- Add a test harness that uses stored local files as inputs in the T0–T2 pipeline.
- Add a UI/CLI tool for running a simple retrieval benchmark using stored corpora and logging results locally (e.g., in OPFS or localStorage).

### 3) Improve UI/UX
- Add optional metadata tagging (e.g., "model", "corpus", "dataset") and search/filter capabilities for stored files.
- Show file sizes, created timestamps, and storage availability (OPFS vs fallback).

### 4) Prepare for “bring your own model” flow
- Add UI to select a stored file to act as the model weights source.
- Wire the worker to accept a local file handle or hash and load the model from OPFS instead of downloading from HuggingFace.

## Notes for future context

- The project is a browser-only ONNX text-generation + Q² quantisation demo.
- Q² kernel is in `src/q2.wat` with a TS wrapper in `src/q2.ts`.
- Tests run via Vitest; browser tests run via Playwright.
- The existing architecture is `app.ts` (UI) + `worker.ts` (inference) + `q2.ts` (Q² logic).
