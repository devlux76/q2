# Parameter Golf strategy

This note outlines how to point Q² at OpenAI's Parameter Golf challenge (train in <10 minutes on 8×H100, 16 MB artifact cap, scored by tokenizer-agnostic val_bpb on FineWeb). It combines the structural quantization stack (§D-2–§D-4) with liquid recurrent blocks inspired by Hasani et al. (LTC / CfC / LIV) and the current leaderboard tactics (int5/6, BigramHash, sliding eval).

## Constraints and current bar
- Submission artifact = compressed model bytes + `train_gpt.py` bytes ≤ 16 000 000.
- Wall-clock training ≤ 10 minutes on 8×H100 SXM; evaluation can be longer but runs on the same hardware profile.
- Metric is `val_bpb` on FineWeb validation; lower is better. Tokenizer is free but must report bytes correctly.
- Current leaderboard (2026-03-20) tops out at ~1.1428 bpb with 10L int5/int6 + BigramHash(10240) + SWA + Muon WD + sliding-window eval.

## Fit for Q²
- Use the equiprobable quaternary thresholds (§D-2.5) so each symbol carries ~2 bits; Gray + Lee structure (§D-2.7, §D-2.8) preserves complement geometry when we quantize weights, activations, and transition keys.
- Mixed-precision oracle from transition density (§D-3.6, §P-17): channels with short runs/high Lee curvature stay at int6/fp16; long-run channels drop to q2.
- Geode progression (§D-4.1, §P-16): start with coarse q2, then refine a small subset of facets (heads/MLP rows) that remain loss-critical after a short probe run.

## Wildberger–Geode structure (answering “the structure itself”)
- Treat each attention head / MLP row as a **facet** in the Geode factorization (§D-4.1): parameter-tie facets that live on the same stratum so depth grows without new parameters; liquid blocks share the same tying map.
- Enforce the **Euler polytope constraint** (§D-4.2) as a budget rule: limit new facets by requiring V−E+F to stay constant when we add heads; this caps parameter growth and keeps the artifact under 16 MB.
- Use **hierarchical unlock**: Stage A trains only the lowest-dimension facets; Stage B unlocks higher-curvature facets whose transition density spikes; Stage C demotes low-curvature facets back to q2 while keeping int6 for the unlocked set.
- Apply **trie sparsity** (§P15) to hashed vocab: prune BigramHash buckets whose transition sequences collapse to shallow tries, freeing bytes for higher-curvature facets.
- During export, **facet ordering** follows Geode traversal so zstd sees long runs (same structure as §D-3.4 block layout), improving compression without changing weights.

## Architecture: liquid + tied + hashed
- **Tokenizer/embeddings**
  - Keep the leaderboard BigramHash(10k) idea but align with Q² by forcing equiprobable symbol usage: hash collisions are re-mapped with the complement involution so the four code points stay balanced (§D-2.5).
  - Tie input/output embeddings (already in baseline) and share them across encoder/decoder halves; keep embeddings at int6 to avoid early entropy loss.
- **Backbone**
  - Replace half of the MLP blocks with a **liquid gated branch**: a compact CfC/LTC cell (3–5 learnable time constants per head) that maintains a per-head state across tokens. This recycles compute across depth and extends effective context without adding attention parameters (mirrors LFM 2.5’s efficiency).
  - Shallow attention (6–8 layers, 4–6 heads, 2 KV heads) plus SmearGate-style mixing; depth is recovered by **parameter tying** (ALBERT-style) + liquid recurrence. This keeps parameter count low for the 16 MB cap while letting us spend wall-clock on more steps.
  - Apply **run-reduction transition keys** to per-channel activations on the fly; use transition density as a regularizer (penalize over-fragmented trajectories) to keep q2 thresholds stable under liquid dynamics.
- **Regularization**
  - Skip-weight vectors and q_gain stay in fp16 (CONTROL_TENSOR guard) but are stored with per-row scaling; everything else targets q2/int5.
  - Use logit softcap (already present) + stochastic depth on liquid branch to stabilize fast compilation.

## Training plan (10-minute budget)
- **Stage A (6–7 min):** train at seq_len 1024 with tied weights and liquid branch active, using a cosine LR warmup→flat→linear warmdown. Compile `zeropower_via_newtonschulz5` and the liquid cell; freeze embeddings for the first 500 steps to stabilize hash collisions.
- **Stage B (2–3 min):** switch to seq_len 2048 with **sliding-window eval (stride 64)** to harvest lower bpb; unfreeze embeddings and apply Muon WD on matrices only. Run a brief QAT pass that drives weights toward equiprobable q2 levels (clip at the τ* thresholds from §D-2.5).
- **Stage C (≤1 min):** structural mixed-precision sweep: promote the densest 5–10% rows (by transition density) to int6, everything else to q2/int5. Capture a checkpoint before compression for ablation.

## Compression and artifact packing
- Design an export path that packs weights into an int8-compatible container while storing the payload as **q2/int5 per-row with 16-bit scales**; keep small control tensors in fp16. Reorder tensors by transition-key order so a downstream compressor (for example, zstd run in the submission pipeline) sees long runs (RLE-friendly).
- Target parameter budget: ≤28 M params → q2 payload ≈ 7 MB; with scales/metadata + code we stay below 15 MB. If BigramHash pushes vocab beyond 10k, cap at 12k and prune rare buckets to stay under budget.
- In the parameter-golf repo, add a round-trip export/import helper (e.g., `final_int8_zlib_roundtrip`) and validate against the tokenizer-agnostic bpb path; reject any run that inflates compressed artifact size past 15.5 MB.

## Execution checklist for the parameter-golf repo
1) Add BigramHash tokenizer variant with complement-aware collision resolution; keep the val_bpb byte accounting unchanged.
2) Implement a **LiquidBlock** (CfC/LTC) that plugs into `train_gpt.py` alongside the existing MLP, with parameter tying across depth (Geode stratum-aware).
3) Add transition-density logging to drive the mixed q2/int6 schedule (P17), plus a tiny QAT head that nudges weights to τ*; unlock higher-dimension facets only when transition density spikes (Geode progression).
4) Swap export to q2/int5 per-row + fp16 control tensors and rely on an external high-ratio compressor (for example, zstd run at a strong setting in the submission toolchain); order tensors by Geode traversal to boost run-length compression.
5) Run the 3-stage schedule above, capture logs for p<0.01 delta over the 1.1428 bpb SOTA.

## Risks and mitigations
- **Liquid stability:** clamp time constants and use per-head RMSNorm to prevent exploding hidden state; fall back to pure MLP if compilation fails.
- **Tokenizer drift:** complement-aware hashing must keep byte accounting identical to baseline; add a regression that compares bytes-per-token vs. the stock SP tokenizer on the val set.
- **Artifact overrun:** if the compressed artifact exceeds 15.5 MB, lower vocab to 8k or increase the int6→q2 demotion threshold until size passes.
