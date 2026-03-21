#!/usr/bin/env bash
# Run Playwright E2E tests locally inside a Docker container.
#
# • Builds the q2-e2e Docker image on first run (one-time ~3 min).
# • Caches node_modules in a named Docker volume so subsequent bun installs
#   are near-instant.
# • Passes /dev/dri to the container for Intel/AMD GPU acceleration when
#   available; falls back to Mesa lavapipe (software Vulkan) otherwise.
# • The pre-installed Playwright browsers in the image are reused — no
#   200 MB chromium download on each run.
#
# Usage:
#   ./e2e/run-local.sh
#   E2E_MODEL=onnx-community/SmolLM2-135M-Instruct E2E_DTYPE=q4 ./e2e/run-local.sh
#
# Rebuild the image after updating the Playwright version in package.json:
#   docker build -t q2-e2e -f e2e/Dockerfile .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="q2-e2e:local"
NM_VOLUME="q2-e2e-node-modules"

# ── Require Docker ────────────────────────────────────────────────────────
if ! command -v docker > /dev/null 2>&1; then
  echo "✖ Docker is not installed or not in PATH." >&2
  echo "  Install Docker Desktop: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

# ── Build image if it does not exist yet ─────────────────────────────────
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "▶ Building E2E Docker image (first-run setup — ~3 min)..."
  docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
fi

# ── GPU passthrough ───────────────────────────────────────────────────────
GPU_FLAGS=()
GPU_AVAILABLE=""
if [ -d /dev/dri ]; then
  GPU_FLAGS+=(--device /dev/dri)
  GPU_AVAILABLE="1"
  echo "▶ /dev/dri found — Intel/AMD GPU passthrough enabled (WebGPU)."
else
  echo "▶ /dev/dri not found — using Mesa lavapipe (software Vulkan fallback)."
fi

# ── Run tests ─────────────────────────────────────────────────────────────
echo "▶ Running E2E tests..."
exec docker run --rm \
  --ipc=host \
  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
  -v "$REPO_ROOT:/app" \
  -v "$NM_VOLUME:/app/node_modules" \
  -w /app \
  -e CI=true \
  -e E2E_GPU_AVAILABLE="${E2E_GPU_AVAILABLE:-$GPU_AVAILABLE}" \
  -e E2E_MODEL="${E2E_MODEL:-}" \
  -e E2E_DTYPE="${E2E_DTYPE:-}" \
  "$IMAGE" \
  sh -c "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 bun install --frozen-lockfile && bun run build && npx playwright test"
