# Parameter Golf: Implementation Roadmap

> **Status**: Ready for implementation
> **Related**: [PARAMETER_GOLF_APPROACH.md](../PARAMETER_GOLF_APPROACH.md)

This document provides tactical implementation details for the Q² Parameter Golf strategy.

---

## Quick Reference

### Key Numbers

- **Target score**: <1.10 bits/byte (current SOTA: 1.1428)
- **Parameter budget**: 16MB = 16,000,000 bytes
- **Training time**: 10 minutes on 8×H100 SXM
- **Effective parameters at int5**: ~25M params

### Architecture Summary

```
Input (BigramHash 10240)
  ↓
Embedding (384 dim, int6) [3.9M params, 2.9MB]
  ↓
12× LTC Blocks (384 dim, MLP 3×, int5) [15M params, 9.4MB]
  ↓
Output (tied with Embedding, int6) [0 params, 0MB]
  ↓
Softmax
```

**Total**: ~19M params, ~12.3MB compressed, 3.7MB headroom

---

## Phase 1: Foundation (Days 1-3)

### Day 1: Environment Setup

**Goal**: Get Parameter Golf baseline running

```bash
# Clone and setup
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy sentencepiece huggingface-hub datasets tqdm

# Download data (10 shards for quick iteration)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Verify baseline runs
RUN_ID=baseline_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=60 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Success criteria**:

- Baseline trains successfully
- Understand `train_gpt.py` structure
- Confirm data loading pipeline

### Day 2: Q² Integration - Weight Quantization

**Goal**: Extend Q² kernel for weight quantization

**Current state**: `src/q2.wat` handles activation quantization (inference-time)

**Needed**: Weight quantization (training-time)

**Implementation**:

1. **Add weight quantization mode to q2.ts**:

```typescript
// src/q2.ts (new function)
export function q2QuantizeWeights(
  weights: Float32Array,
  dim: number,
  precision: 5 | 6 | 8 = 6
): Uint8Array {
  // Compute per-tensor or per-channel thresholds
  const tau = computeEquiprobableThreshold(weights, dim);

  // Quantize to {A, B, C, D} = {0, 1, 2, 3}
  const quantized = new Uint8Array(Math.ceil(weights.length * precision / 8));

  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    let sym: number;

    if (w <= -tau) sym = 0;      // A: strong negative
    else if (w <= 0) sym = 1;    // B: weak negative
    else if (w <= tau) sym = 2;  // C: weak positive
    else sym = 3;                // D: strong positive

    // Gray encode
    const gray = sym ^ (sym >> 1);

    // Pack based on precision
    packSymbol(quantized, i, gray, precision);
  }

  return quantized;
}
```

2. **Add analytical threshold computation** (§D-4.4):

```typescript
function computeEquiprobableThreshold(
  weights: Float32Array,
  dim: number
): number {
  // For Gaussian approximation: τ* = Φ^(-1)(3/4) / √n
  const invCDF_75 = 0.6745; // Φ^(-1)(0.75)
  const n = dim;
  return invCDF_75 / Math.sqrt(n);
}

// For non-Gaussian: use empirical quartiles
function computeEmpiricalThreshold(weights: Float32Array): number {
  const sorted = Float32Array.from(weights).sort();
  const q25 = sorted[Math.floor(sorted.length * 0.25)];
  const q75 = sorted[Math.floor(sorted.length * 0.75)];
  return (Math.abs(q25) + Math.abs(q75)) / 2;
}
```

3. **Test on toy model**:

```typescript
// test/weight-quantization.test.ts
import { q2QuantizeWeights } from '../src/q2';

test('weight quantization preserves distribution', () => {
  const weights = new Float32Array(1024);
  // Generate Gaussian N(0, 1/32)
  for (let i = 0; i < weights.length; i++) {
    weights[i] = gaussianRandom(0, 1/32);
  }

  const quantized = q2QuantizeWeights(weights, 32, 6);

  // Verify: ~25% in each of {A, B, C, D}
  const counts = countSymbols(quantized, 6);
  expect(counts.A).toBeCloseTo(256, 30); // ±30 for variance
  expect(counts.B).toBeCloseTo(256, 30);
  expect(counts.C).toBeCloseTo(256, 30);
  expect(counts.D).toBeCloseTo(256, 30);
});
```

**Success criteria**:

- Weight quantization function works
- Equiprobable distribution verified
- Compression ratio: 4 weights/byte (int6) or 3.2 weights/byte (int5)

### Day 3: PyTorch QAT Integration

**Goal**: Hook Q² quantization into PyTorch training

**Implementation**:

```python
# q2_pytorch/quantize.py
import torch
import torch.nn as nn

class Q2QuantizeWeight(torch.autograd.Function):
    """Quantization-Aware Training for Q² weights"""

    @staticmethod
    def forward(ctx, weight, tau, precision=6):
        # Quantize to {0, 1, 2, 3}
        sym = torch.zeros_like(weight, dtype=torch.long)
        sym[weight <= -tau] = 0  # A
        sym[(weight > -tau) & (weight <= 0)] = 1  # B
        sym[(weight > 0) & (weight <= tau)] = 2  # C
        sym[weight > tau] = 3  # D

        # Dequantize to {-1.5, -0.5, +0.5, +1.5} * tau (centered)
        dequant_table = torch.tensor(
            [-1.5, -0.5, 0.5, 1.5], dtype=weight.dtype, device=weight.device
        )
        weight_q = dequant_table[sym] * tau

        ctx.save_for_backward(weight, weight_q, tau)
        return weight_q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator (STE)
        weight, weight_q, tau = ctx.saved_tensors
        grad_weight = grad_output.clone()

        # Gradient clipping at quantization boundaries
        # (optional refinement)
        return grad_weight, None, None


class Q2Linear(nn.Linear):
    """Linear layer with Q² quantization"""

    def __init__(self, in_features, out_features, bias=True, precision=6):
        super().__init__(in_features, out_features, bias)
        self.precision = precision
        self.register_buffer('tau', torch.tensor(0.6745 / (in_features ** 0.5)))

    def forward(self, x):
        # Quantize weights on-the-fly during training
        if self.training:
            weight_q = Q2QuantizeWeight.apply(self.weight, self.tau, self.precision)
        else:
            # Use pre-quantized weights during inference
            weight_q = self.weight_quantized

        return nn.functional.linear(x, weight_q, self.bias)

    def quantize_weights(self):
        """Finalize quantization (call before export)"""
        with torch.no_grad():
            self.weight_quantized = Q2QuantizeWeight.apply(
                self.weight, self.tau, self.precision
            )
```

**Test**:

```python
# Test QAT on 2-layer MLP
model = nn.Sequential(
    Q2Linear(768, 3072, precision=6),
    nn.GELU(),
    Q2Linear(3072, 768, precision=6),
)

# Train for a few steps
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(100):
    x = torch.randn(32, 768)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Quantize for export
for layer in model:
    if isinstance(layer, Q2Linear):
        layer.quantize_weights()
```

**Success criteria**:

- QAT trains without errors
- Loss converges (even on random data)
- Weight quantization called successfully

---

## Phase 2: Core Architecture (Days 4-10)

### Day 4-5: LTC Block Implementation

**Goal**: Implement Closed-form Continuous-time (CfC) cells

**Background**: LTC/CfC from Hasani et al. (2021, 2023)

**Key equations**:

$$\frac{dx}{dt} = f(x, u, \theta)$$

where $f$ is a learned ODE. CfC uses closed-form solution:

$$x_{t+1} = x_t + \int_t^{t+1} f(x(\tau), u, \theta) d\tau$$

**Implementation** (adapted from `ncps` library):

```python
# q2_pytorch/cfc.py
import torch
import torch.nn as nn

class CfCCell(nn.Module):
    """Closed-form Continuous-time Cell (Hasani et al. 2023)"""

    def __init__(self, input_size, hidden_size, sparsity=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Wiring: sparse connectivity (NCP-style)
        # Only connect sparsity% of neuron pairs
        self.register_buffer(
            'wiring_mask',
            self._create_sparse_mask(hidden_size, sparsity)
        )

        # ODE parameters: f(x) = -a*x + b*σ(Wx + Uu + bias)
        self.w_tau = nn.Parameter(torch.randn(hidden_size))  # Time constants
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, hidden_size)

        # Activation
        self.activation = nn.Tanh()

        # Apply sparsity mask to recurrent weights
        self.w_rec.weight.data *= self.wiring_mask

    def _create_sparse_mask(self, size, sparsity):
        """Create sparse connectivity mask"""
        mask = torch.rand(size, size) < sparsity
        mask.fill_diagonal_(True)  # Always self-connect
        return mask.float()

    def forward(self, x, h):
        """
        Args:
            x: input (batch, input_size)
            h: hidden state (batch, hidden_size)
        Returns:
            h_next: next hidden state (batch, hidden_size)
        """
        # Compute ODE right-hand side
        u_in = self.w_in(x)
        u_rec = self.w_rec(h) * self.wiring_mask.unsqueeze(0)
        u = self.activation(u_in + u_rec)

        # Closed-form solution: Euler integration with learned time constants
        tau = torch.sigmoid(self.w_tau)  # Time constants in (0, 1)
        h_next = (1 - tau) * h + tau * u

        return h_next


class LTCBlock(nn.Module):
    """LTC block replacing transformer attention"""

    def __init__(self, dim, mlp_ratio=3.0, sparsity=0.5):
        super().__init__()
        self.dim = dim

        # CfC cell
        self.cfc = CfCCell(dim, dim, sparsity=sparsity)

        # MLP (following leaderboard: 3× expansion)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            SmearGeLU(),  # Leaderboard-proven activation
            nn.Linear(mlp_hidden, dim),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, h=None):
        """
        Args:
            x: input (batch, seq_len, dim)
            h: optional hidden state from previous sequence
        Returns:
            output: (batch, seq_len, dim)
            h_last: final hidden state for next sequence
        """
        batch, seq_len, dim = x.shape

        # Initialize hidden state
        if h is None:
            h = torch.zeros(batch, dim, device=x.device)

        # Process sequence with CfC
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.cfc(self.ln1(x_t), h)
            outputs.append(h)

        x = torch.stack(outputs, dim=1)  # (batch, seq_len, dim)

        # Add & Norm
        x = x + torch.stack(outputs, dim=1)

        # MLP
        x = x + self.mlp(self.ln2(x))

        return x, h


class SmearGeLU(nn.Module):
    """SmearGate activation (from leaderboard)"""
    def forward(self, x):
        # SmearGate: smooth variant of GeLU
        return x * torch.sigmoid(1.702 * x)
```

**Test**:

```python
# Test LTC block
block = LTCBlock(dim=384, mlp_ratio=3.0)

x = torch.randn(32, 128, 384)  # batch=32, seq=128, dim=384
output, h = block(x)

assert output.shape == (32, 128, 384)
assert h.shape == (32, 384)

# Test sequential processing
output2, h2 = block(x, h)  # Continue from previous state
```

**Success criteria**:

- LTC block processes sequences
- Hidden state propagates correctly
- Gradients flow (test with dummy loss)

### Day 6-7: Full Model Architecture

**Goal**: Assemble full Q²-LTC model

```python
# q2_pytorch/model.py
import torch
import torch.nn as nn
from .cfc import LTCBlock
from .quantize import Q2Linear

class Q2ParameterGolfModel(nn.Module):
    """Q² model for Parameter Golf challenge"""

    def __init__(
        self,
        vocab_size=10240,  # BigramHash
        dim=384,
        n_layers=12,
        mlp_ratio=3.0,
        cfc_sparsity=0.5,
        precision=6,  # int6 default, can mix with int5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        # Embedding (int6 as proven on leaderboard)
        self.embed = nn.Embedding(vocab_size, dim)

        # LTC blocks
        self.blocks = nn.ModuleList([
            LTCBlock(dim, mlp_ratio, cfc_sparsity)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

        # Output projection (tied with embedding)
        # Will be set after embed is initialized
        self.output = None

    def forward(self, idx, hidden_states=None):
        """
        Args:
            idx: input token indices (batch, seq_len)
            hidden_states: optional list of hidden states from previous batch
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_hidden_states: list of hidden states for next batch
        """
        x = self.embed(idx)  # (batch, seq_len, dim)

        if hidden_states is None:
            hidden_states = [None] * self.n_layers

        new_hidden_states = []
        for i, block in enumerate(self.blocks):
            x, h = block(x, hidden_states[i])
            new_hidden_states.append(h)

        x = self.ln_f(x)

        # Output projection (tied)
        if self.output is None:
            # Tie output with embedding
            self.output = nn.Linear(self.dim, self.vocab_size, bias=False)
            self.output.weight = self.embed.weight

        logits = self.output(x)

        return logits, new_hidden_states

    def quantize_model(self, precision_map=None):
        """
        Quantize all weights for export

        Args:
            precision_map: dict mapping layer name to precision (5, 6, or 8)
                          If None, use int6 for all
        """
        if precision_map is None:
            precision_map = {}

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Determine precision for this layer
                precision = precision_map.get(name, 6)

                # Quantize weights
                # (In practice, replace nn.Linear with Q2Linear)
                pass  # TODO: implement weight replacement

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def estimate_compressed_size(self, precision_map=None):
        """
        Estimate compressed size in bytes

        Args:
            precision_map: dict mapping layer types to average precision
        """
        if precision_map is None:
            precision_map = {'embed': 6, 'ltc': 5, 'mlp': 5, 'output': 0}

        total_bytes = 0

        # Embedding (int6)
        embed_params = self.vocab_size * self.dim
        total_bytes += embed_params * 6 / 8

        # LTC blocks (mixed int5/int6)
        ltc_params = 0
        for block in self.blocks:
            ltc_params += sum(p.numel() for p in block.parameters())

        total_bytes += ltc_params * 5 / 8  # Conservative int5

        # Output (tied, no extra params)

        # Add zstd compression factor (typical: 0.8×)
        total_bytes *= 0.8

        return total_bytes
```

**Test**:

```python
# Test full model
model = Q2ParameterGolfModel(
    vocab_size=10240,
    dim=384,
    n_layers=12,
    mlp_ratio=3.0,
)

print(f"Total parameters: {model.count_parameters():,}")
print(f"Estimated size: {model.estimate_compressed_size() / 1e6:.2f} MB")

# Forward pass
idx = torch.randint(0, 10240, (32, 128))
logits, hidden = model(idx)

assert logits.shape == (32, 128, 10240)
```

**Success criteria**:

- Model initializes
- Parameters < 25M
- Estimated compressed size < 13MB (headroom for optimization)
- Forward pass works

### Day 8-10: Training Loop Integration

**Goal**: Integrate Q² model into Parameter Golf training script

**Create** `q2_train_gpt.py` (fork of `train_gpt.py`):

```python
# Key modifications to train_gpt.py:

# 1. Replace GPT model with Q2ParameterGolfModel
from q2_pytorch.model import Q2ParameterGolfModel

model = Q2ParameterGolfModel(
    vocab_size=args.vocab_size,
    dim=384,
    n_layers=12,
    mlp_ratio=3.0,
).to(device)

# 2. Use Muon optimizer (proven on leaderboard)
from muon import Muon  # Custom optimizer

optimizer = Muon(
    model.parameters(),
    lr=0.01,
    momentum=0.99,
    weight_decay=0.04,
)

# 3. Three-phase training schedule
def get_precision_schedule(step, total_steps):
    """
    Phase 1 (0-30%): int8 (coarse learning)
    Phase 2 (30-80%): int6/int5 mixed (refinement)
    Phase 3 (80-100%): int5 (final)
    """
    if step < total_steps * 0.3:
        return 8
    elif step < total_steps * 0.8:
        return 6
    else:
        return 5

# 4. Geode-guided progressive quantization
def quantize_progressively(model, current_layer, precision):
    """Quantize layers 0..current_layer, leave rest at higher precision"""
    for i, block in enumerate(model.blocks):
        if i <= current_layer:
            # Quantize this block
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    # Apply Q² quantization
                    pass

# 5. Sliding window evaluation (proven on leaderboard)
def evaluate_sliding_window(model, data, window=2048, stride=64):
    """Evaluate with sliding window for longer effective context"""
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(data) - window, stride):
        chunk = data[i:i+window]
        logits, _ = model(chunk[:-1])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            chunk[1:].view(-1)
        )
        total_loss += loss.item() * (window - 1)
        total_tokens += (window - 1)

    return total_loss / total_tokens
```

**Success criteria**:

- Training starts successfully
- GPU memory fits on H100 (80GB)
- Training completes in ~10 minutes
- Model compresses to <16MB

---

## Phase 3: Optimization (Days 11-20)

### Day 11-13: Hyperparameter Tuning

**Key hyperparameters to tune**:

1. **Model architecture**:
   - `dim`: 320, 384, 448
   - `n_layers`: 10, 12, 14
   - `mlp_ratio`: 2.5, 3.0, 3.5
   - `cfc_sparsity`: 0.4, 0.5, 0.6

2. **Training**:
   - `lr`: 0.008, 0.01, 0.012
   - `weight_decay`: 0.03, 0.04, 0.05
   - `batch_size`: Varies based on context length
   - `seq_length`: 2048, 3072, 4096

3. **Quantization**:
   - `int5_ratio`: 0.6, 0.7, 0.8 (% of layers at int5)
   - `embed_precision`: 6, 7
   - `threshold_method`: 'analytical', 'empirical', 'learned'

**Tuning strategy**:

```python
# Grid search (if time permits)
configs = [
    {'dim': 384, 'n_layers': 12, 'mlp_ratio': 3.0},
    {'dim': 320, 'n_layers': 14, 'mlp_ratio': 3.0},
    {'dim': 448, 'n_layers': 10, 'mlp_ratio': 2.5},
]

results = []
for config in configs:
    # Train for 10 min
    score = train_and_evaluate(**config)
    results.append((config, score))

# Pick best
best_config = min(results, key=lambda x: x[1])
```

### Day 14-16: Ablation Studies

**Questions to answer**:

1. **Does LTC beat attention?**
   - Train Q²-Attention baseline
   - Compare to Q²-LTC at same parameter count
   - Expected: LTC wins by 0.01-0.02 bpb

2. **Does structural quantization beat reconstruction?**
   - Implement GPTQ-style reconstruction quantization
   - Compare to Q² structural quantization
   - Expected: Q² wins by 0.02-0.03 bpb

3. **Does Geode-guided training help?**
   - Train flat (standard end-to-end)
   - Train with progressive quantization
   - Expected: Progressive wins by 0.01 bpb

4. **What's the optimal int5/int6 ratio?**
   - Try 50%, 70%, 90% int5
   - Expected: 70% optimal (balance compression & quality)

### Day 17-20: Final Optimization

**Last mile improvements**:

1. **Gradient accumulation tuning**
   - Find optimal batch size vs. # steps trade-off
   - Larger batches → fewer steps but better gradients

2. **Learning rate warmup/cooldown**
   - Tune warmup duration
   - Tune final LR (for SWA)

3. **Tokenizer optimization**
   - BigramHash parameters
   - Character vs. byte-level encoding

4. **Compression pipeline**
   - Test zstd compression levels (20, 21, 22)
   - Ensure final artifact < 16MB

**Target by end of Phase 3**: **1.10-1.12 bpb** consistently

---

## Phase 4: Submission (Days 21-25)

### Day 21-23: Final Runs

**Reproducibility testing**:

```bash
# Run 5 times with different seeds
for seed in 1 2 3 4 5; do
    SEED=$seed \
    RUN_ID=q2_ltc_final_seed${seed} \
    torchrun --standalone --nproc_per_node=8 q2_train_gpt.py
done

# Compute mean and std
python analyze_runs.py
```

**Statistical significance**:

- Need to beat 1.1428 by ≥0.005 nats (challenge requirement)
- Target: 1.10 mean, σ < 0.005
- p < 0.01 via t-test

### Day 24: Prepare Submission

**Required files** (per Parameter Golf guidelines):

1. **README.md**:
   ```markdown
   # Q² + Liquid Time Constants for Parameter Golf

   ## Summary
   This submission combines Q² structural quantization with Liquid Time
   Constant (LTC) blocks to achieve state-of-the-art compression on
   FineWeb validation.

   **Score**: 1.10 bpb (avg over 5 runs, σ=0.004)

   ## Key Innovations
   1. Structural quantization (Q²) instead of reconstruction quantization
   2. LTC blocks instead of attention (linear complexity)
   3. Geode-guided progressive quantization
   4. Mixed int5/int6 precision guided by hyper-Catalan framework

   ## Reproducibility
   ```bash
   # Clone repo
   git clone https://github.com/<user>/parameter-golf-q2
   cd parameter-golf-q2

   # Setup
   pip install -r requirements.txt
   python3 data/cached_challenge_fineweb.py --variant sp1024

   # Train
   torchrun --standalone --nproc_per_node=8 q2_train_gpt.py
   ```

   ## Authors
   Q² Project Team
   ```

2. **submission.json**:
   ```json
   {
     "name": "Q² Project",
     "github_id": "devlux76",
     "val_bpb": 1.10,
     "runs": [1.098, 1.102, 1.099, 1.101, 1.100],
     "date": "2026-04-15",
     "description": "Q² structural quantization + LTC blocks",
     "innovations": [
       "Structural quantization (Lee metric on Z4)",
       "Liquid Time Constant blocks (linear complexity)",
       "Geode-guided progressive quantization",
       "Hyper-Catalan mixed-precision allocation"
     ]
   }
   ```

3. **Train logs** (5 runs)

4. **q2_train_gpt.py** (full script)

5. **requirements.txt**:
   ```
   torch>=2.0.0
   numpy
   sentencepiece
   ```

### Day 25: Submit

**Submission process**:

1. Create folder: `records/track_10min_16mb/2026-04-15_Q2_LTC_Structural_Quant/`

2. Add all required files

3. Create PR to `openai/parameter-golf`:
   ```bash
   git checkout -b q2-ltc-submission
   git add records/track_10min_16mb/2026-04-15_Q2_LTC_Structural_Quant/
   git commit -m "Q² + LTC: 1.10 bpb via structural quantization"
   git push origin q2-ltc-submission
   ```

4. Create PR on GitHub

5. Monitor for CI verification

---

## Risk Mitigation Checklist

### Before Day 10 (Architecture Lock)

- [ ] Q² weight quantization works
- [ ] LTC blocks train successfully
- [ ] Full model fits in memory
- [ ] Estimated size < 14MB (2MB headroom)

**If blocked**: Fall back to standard attention, Q² still competitive

### Before Day 15 (Optimization Lock)

- [ ] Training completes in < 10 min
- [ ] Score < 1.15 bpb (competitive)
- [ ] Ablations show each component helps

**If blocked**: Simplify architecture (fewer layers or narrower)

### Before Day 20 (Submission Lock)

- [ ] Score < 1.12 bpb (beats SOTA)
- [ ] Reproducible (3+ runs within 0.01 bpb)
- [ ] Submission files ready

**If blocked**: Submit as "non-record" with detailed analysis of approach

---

## Success Metrics by Phase

| Phase | End Date | Metric | Target | Stretch |
|-------|----------|--------|--------|---------|
| 1. Foundation | Day 3 | Code works | ✓ Pass tests | Early finish |
| 2. Core Arch | Day 10 | Trains | ✓ < 10 min | < 8 min |
| 3. Optimization | Day 20 | Score | < 1.12 bpb | < 1.10 bpb |
| 4. Submission | Day 25 | PR | Submitted | Accepted |

---

## Daily Standup Template

**What I did yesterday**:

- [Completed tasks]

**What I'm doing today**:

- [Planned tasks]

**Blockers**:

- [Any issues]

**Metrics**:

- Training time: X min
- Current score: X.XX bpb
- Model size: X.X MB

---

## Resources

### External Dependencies

- **PyTorch**: 2.0+ (H100 support)
- **ncps**: For LTC/CfC reference implementation
  ```bash
  pip install ncps
  ```
- **Muon optimizer**: Custom (include in repo)

### Compute

- **Development**: 1×H100 on RunPod (~$2-3/hour)
  - Budget: $300-500 for 2-3 weeks
- **Final runs**: 8×H100 SXM
  - Budget: $200 for 10-15 runs

**Total estimated cost**: $500-700

### Team

- **Developer** (1 person, full-time equivalent):
  - PyTorch experience (required)
  - WASM/low-level (helpful)
  - ML research (helpful)

---

## Next Actions

**Immediate** (Today):

1. Set up RunPod account and spin up 1×H100
2. Clone `parameter-golf` repo and verify baseline
3. Start Q² weight quantization implementation

**This Week**:

1. Complete Phase 1 (Foundation)
2. Begin Phase 2 (Core Architecture)
3. First full model training run

**By End of Month**:

1. Complete all phases
2. Submit to Parameter Golf
3. Publish results

---

**Document Status**: Living document, update as implementation progresses

**Last Updated**: 2026-03-21

**Owner**: Q² Project
