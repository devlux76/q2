# Q² Benchmark Results

> **Related documents:** [DESIGN.md](DESIGN.md) · [TESTING.md](TESTING.md) · [PREDICTIONS.md](PREDICTIONS.md)

---

## T1 — Null Baselines

### P10: 64-bit key collision rate — PASS

0.00% collisions (1000/1000 unique keys). Confirms the 64-bit transition key has
sufficient entropy to avoid collisions over a small synthetic corpus, consistent
with the uniform-baseline prediction in P10.

---

### Null: uniform Z₄ symbol frequency — investigation

#### Original result (FAIL): χ²=3127.86 (threshold=7.81)

The benchmark originally generated random input vectors as:

```ts
for (let i = 0; i < n; i++) vec[i] = Math.random() * 2 - 1;
```

This produces components drawn from Uniform[-1, 1]. After L2 normalisation of a
128-dimensional vector, the squared norm concentrates tightly around
E[‖u‖²] = n/3 = 128/3, so each normalised component follows approximately:

```
v_i ~ u_i / sqrt(n/3)  ->  Uniform[-sqrt(3/n), sqrt(3/n)]  ~  Uniform[-0.153, 0.153]
```

The quantisation threshold is τ* = Φ⁻¹(¾)/√n ≈ 0.6745/√128 ≈ 0.0596, which
is designed to be the **Gaussian quartile** — it places exactly 25% of N(0, 1/n)
probability mass in each of the four symbol cells (DESIGN.md §2.4).

For uniform marginals, however, τ* sits at ≈ 69.5% of the distribution (not 75%),
producing systematically skewed symbol probabilities:

| Symbol | Designed P | Actual P (uniform input) |
|--------|-----------|--------------------------|
| A (strong−) | 25% | ≈ 30.5% |
| B (weak−)   | 25% | ≈ 19.5% |
| C (weak+)   | 25% | ≈ 19.5% |
| D (strong+) | 25% | ≈ 30.5% |

With 500 trials × 128 dimensions = 64,000 total symbols, the predicted χ² is:

```
chi2 = 4 * (3520^2 / 16000) ~ 3098
```

This matches the observed 3127.86 to within rounding of the approximated marginal
distribution — the failure was **theoretically guaranteed** by the distribution
mismatch, not an encoder defect.

#### Why this is informative

The large χ² confirms that the encoder threshold **is** doing exactly what the
design claims: τ* is at the Gaussian quartile, making it sensitive to the input
distribution. A broken or random encoder would not produce such a predictable,
analytically-computable deviation. The failure is a precise measurement, not noise.

This also validates why DESIGN.md §2.4 specifies empirical calibration in
production (using the empirical 25th/75th percentiles of actual model activations):
the fixed τ* only achieves maximum symbol entropy when the input marginals are
Gaussian, which real embedding models approximate but do not guarantee exactly.

#### Fix

The benchmark was corrected to generate pre-normalisation components from N(0, 1)
using Box-Muller, matching the Gaussian assumption under which τ* was derived:

```ts
// Box-Muller: pairs of uniform samples -> standard normal pairs
for (let i = 0; i < n; i += 2) {
  const u1 = Math.random(), u2 = Math.random();
  const r = Math.sqrt(-2 * Math.log(u1));
  vec[i] = r * Math.cos(2 * Math.PI * u2);
  if (i + 1 < n) vec[i + 1] = r * Math.sin(2 * Math.PI * u2);
}
```

With Gaussian input the symbol distribution is equiprobable by construction, and
χ² ≪ 7.81 (passes).
