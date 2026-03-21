# syntax=docker/dockerfile:1
#
# E2E local test image for Q².
#
# Built on the official Playwright Ubuntu Jammy image — Chromium and all of
# its system-level dependencies are pre-installed and version-matched.
# Mesa Vulkan drivers are layered on top so that Chromium can use WebGPU via:
#   • Intel ANV  (Intel integrated / discrete GPU)
#   • AMD RADV   (AMD GPU via /dev/dri passthrough)
#   • lavapipe   (Mesa software Vulkan — unconditional fallback, no GPU needed)
#
# Build:  docker build -t q2-e2e -f e2e/Dockerfile .
# Run:    bun run test:e2e:local   (or directly: ./e2e/run-local.sh)
#
# Update this tag when you bump @playwright/test in package.json.
# Available tags: https://mcr.microsoft.com/v2/playwright/tags/list
FROM mcr.microsoft.com/playwright:v1.50.1-jammy

# ── Vulkan support ────────────────────────────────────────────────────────
# mesa-vulkan-drivers installs Intel ANV, AMD RADV, and lavapipe in one shot.
# libvulkan1 is the Vulkan loader (required by both drivers and Chromium).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# ── Bun ───────────────────────────────────────────────────────────────────
# Bun is the project's build tool and package manager.
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

WORKDIR /app
