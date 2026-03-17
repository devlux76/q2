# q2
Quaternary Quantization

Q2 starts with quaternary quantization of a local model's own native embeddings. This produces something of a fingerprint for the semantic geometry the model is currently evaluating.

This geometry is a product of human language itself. Therefore, we propose that mapping the geometry will produce faster and more accurate embeddings and we believe it most likely solves the incommensurability problem of vector similarity search.

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

## Deploy (GitHub Pages)

This project is a static browser app (HTML + JS bundle). To host it on GitHub Pages, build the bundle and publish the `dist/` output as the Pages site.

1. **Build** (produces `dist/app.js`):

```bash
bun run build
```

2. **Copy the static entrypoints into `dist/`** so `index.html` can reference `dist/app.js` correctly:

```bash
cp index.html style.css dist/
```

3. **Publish `dist/` to GitHub Pages** (push to the `gh-pages` branch):

```bash
git add dist/index.html dist/style.css
git commit -m "chore: build for gh-pages"
# Push dist/ as the root of the gh-pages branch
git subtree push --prefix dist origin gh-pages
```

4. In your repo settings, enable GitHub Pages and set the source to the `gh-pages` branch (root).

### Optional: Auto deploy on push to main

This repository includes a GitHub Actions workflow (`.github/workflows/gh-pages.yml`) that automatically builds and publishes `dist/` to the `gh-pages` branch whenever you push to `main`.

### Optional local sanity check

To verify the built site loads before deploying, serve `dist/` locally with a static server (this is just for local testing):

```bash
npx serve dist
```

Then open the URL it prints (e.g. `http://localhost:3000`).
