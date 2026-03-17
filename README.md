# q2
Quaternary Quantization

## Screenshots

### Loading Screen
The app displays a progress card while downloading and caching the ~1.2 GB ONNX model weights.

![Load screen — model downloading](https://github.com/user-attachments/assets/3a5f66ad-a352-42ae-ad1a-a523cedb98a9)

### Chat Interface (empty)
Once the model is ready, the full chat interface appears with the generation settings sidebar.

![Chat interface — empty state](https://github.com/user-attachments/assets/68b3595d-5c94-4923-bbd6-f106ea759503)

### Chat Interface (conversation)
During and after a conversation the sidebar also shows the **Last LIV layer embeddings** panel — a heat-map of the raw activations fed into the Q² quantization kernel.

![Chat interface — active conversation with embedding panel](https://github.com/user-attachments/assets/555cc7ee-012d-4e1b-8db7-14a164e4f462)

## Setup

1. Install [Bun](https://bun.sh/) (required).
2. Install dependencies:

```bash
bun install
```

## Scripts

- **Build** (production bundle):

```bash
bun run build
```

- **Dev** (watch mode):

```bash
bun run dev
```

- **Typecheck** (TypeScript):

```bash
bun run typecheck
```

- **Test** (unit tests + coverage):

```bash
bun run test
```

- **Browser tests** (runs tests in a real browser via Playwright):

```bash
bun run test:browser
```

> If Playwright browsers are not yet installed, run:
>
> ```bash
> bun x playwright install
> ```
