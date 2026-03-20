# Q² Geometry Animation Script

## Narrative Arc

**From broken L2 sphere → rational L1 octahedron → Z4 fingerprint**

The animation tells one coherent story in nine acts. Each phase hands off visually
to the next. No phase is decorative; every one advances the mathematical argument.

---

## Color Palette

| Symbol | Hex       | Z4 value | Role              |
|--------|-----------|----------|-------------------|
| CYAN   | `#00f0ff` | A = 0    | Primary / axes    |
| MAGENTA| `#ff00aa` | B = 1    | Complement of D   |
| LIME   | `#aaff00` | C = 2    | Complement of A   |
| ORANGE | `#ffaa00` | D = 3    | Complement of B   |
| WHITE  | `#ffffff` | —        | Labels / edges    |
| BG     | `#0a0a1f` | —        | Background        |

Complement pairs: **A ↔ C** (0 ↔ 2)  and  **B ↔ D** (1 ↔ 3)

---

## Phase Schedule

| # | Name             | Frames | Cumulative |
|---|------------------|--------|------------|
| 1 | `seed`           | 50     | 50         |
| 2 | `l2_project`     | 75     | 125        |
| 3 | `pi4_declare`    | 60     | 185        |
| 4 | `sphere_to_octa` | 80     | 265        |
| 5 | `z4_label`       | 65     | 330        |
| 6 | `grid_lines`     | 65     | 395        |
| 7 | `complement`     | 55     | 450        |
| 8 | `gray_code`      | 55     | 505        |
| 9 | `fingerprint`    | 65     | 570        |

**Total: 570 frames · 15 fps · ~38 seconds**

---

## Phase Details

### ① Seed (frames 0–49)

**Visual**
- Words from `"I can believe 10 impossible things before breakfast"` fade in one by
  one, scattered at random positions inside the unit cube.
- Faint cube wireframe (WHITE, α=0.15).

**Camera**: Start high (elev≈70°), tilt downward as phase progresses.

**Overlay title**: "① Text Seed"
**Subtitle**: the full phrase
**Note**: "Raw token positions in high-dimensional embedding space"

---

### ② L2 Project (frames 50–124)

**Visual**
- Cyan L2 sphere wireframe fades in (α: 0.08 → 0.25).
- Each word point slides radially from its scatter position to the unit sphere surface.
  Motion trail: a short line from old position to new.
- Cube wireframe stays faint.

**Camera**: Slow rotation, elev≈30°.

**Math label**: `‖x‖₂ = 1`

**Overlay title**: "② L2 Normalise → Unit Sphere"
**Subtitle**: `S^{n-1} = { x ∈ ℝⁿ : ‖x‖₂ = 1 }`
**Note**: "Problem: O(n) rotational freedom ⟹ inter-model incommensurability"

---

### ③ π = 4 (frames 125–184)

**Visual**
- Sphere wireframe stays visible (MAGENTA, α=0.15).
- Large "π = 4" (LIME, fontsize 52) fades in at center.
- Below it, after a short delay: "Switch from L² to L¹ — the one impossible belief
  that dissolves the problem" (ORANGE, italic).

**Camera**: Slow pull-back, elev stable.

**Overlay title**: "③ The π = 4 Insight"
**Subtitle**: "Hypersphere incommensurability is unsolvable in L² — so leave L²"
**Note**: "L¹ cross-polytope has rational geometry and exact π = 4"

---

### ④ Sphere → Octahedron (frames 185–264)  ← KEY VISUAL

**Visual**
This is the centrepiece of the animation.

- A theta/phi mesh sampled on the L2 sphere morphs continuously to the L1 unit
  surface (the octahedron) by blending:

      P(t) = (1 - t)·P_sphere + t·(P_sphere / ‖P_sphere‖₁)

  The mesh lines straighten from curves into flat triangular facet lines.

- Simultaneously, each word point slides from its sphere position to its
  L1-projected position.

- After the halfway point (t > 0.5): octahedron coloured faces fade in with
  ABCD colours, solidifying the geometry.

**Camera**: Rotate continuously so viewer sees all faces of the morph.

**Math labels**: `‖x‖₂ = 1  →  ‖x‖₁ = 1`

**Overlay title**: "④ Sphere → Cross-Polytope (Octahedron)"
**Subtitle**: `‖x‖₂=1  ⟶  ‖x‖₁=1`
**Note**: "L¹ unit ball = octahedron — 8 flat triangular faces, 6 vertices on axes, ρ = 4"

---

### ⑤ Z4 Label (frames 265–329)

**Visual**
- Clean octahedron with ABCD-coloured faces, full opacity.
- Word points resting on octahedron surface.
- ABCD labels fade in above the four upper-face quadrants:
    - A (0) CYAN top-right
    - B (1) MAGENTA top-left
    - C (2) LIME bottom-left
    - D (3) ORANGE bottom-right
- After 60% progress, glowing sequence appears:
    `A(0) → B(1) → C(2) → D(3) → A(0)  [mod 4]`
  followed by `effective ρ = 4  ·  Lee metric`

**Camera**: Stable 3/4 view, slow rotation.

**Overlay title**: "⑤ Z₄ Quantisation on Octahedron"
**Subtitle**: `ℤ/4ℤ : four regions, Lee metric d_L(a,b) = min(|a−b|, 4−|a−b|)`
**Note**: "Each face carries one quaternary symbol: A=0  B=1  C=2  D=3"

---

### ⑥ Grid Lines (frames 330–394)

**Visual**
- Cube wireframe fades in more strongly (α: 0.25 → 0.40).
- Six dashed grid divider lines appear, partitioning the cube into the ABCD
  quadrant regions (same as existing `draw_grid_lines`).
- Octahedron stays visible inside.
- Word points rest at their quantised octahedron positions.
- ABCD quadrant labels on cube top face.

**Camera**: Slightly higher elev to see grid structure.

**Overlay title**: "⑥ Grid Quadrant Lines"
**Subtitle**: "Four dividers partition the cube into {A, B, C, D} quantisation regions"
**Note**: "Each word point lands in exactly one quaternary region"

---

### ⑦ Complement Involution (frames 395–449)

**Visual**
- Octahedron face colours pulse and swap:
    - A (CYAN) ↔ C (LIME) faces cross-fade back and forth
    - B (MAGENTA) ↔ D (ORANGE) faces cross-fade back and forth
  using a sinusoidal lerp so the swap oscillates visibly.
- Text appears: `θ(x) = x + 2  (mod 4)` (WHITE, bold)
- Below: `A ↔ C    B ↔ D` (LIME)

**Camera**: Stable.

**Overlay title**: "⑦ Complement Involution"
**Subtitle**: `θ : x ↦ x + 2 (mod 4)  — antipodal face swap`
**Note**: "Self-inverse: θ(θ(x)) = x  ·  maps each symbol to its antipodal partner"

---

### ⑧ Gray Code (frames 450–504)

**Visual**
- Octahedron visible at lower alpha (background reference).
- Binary labels appear in screen-space near each quadrant:
    - A = `00` (CYAN)
    - B = `01` (MAGENTA)
    - C = `11` (LIME)
    - D = `10` (ORANGE)
- After 50% progress, the Gray path traces:
    `00 → 01 → 11 → 10 → 00`
- Note: "Adjacent codes differ by exactly 1 bit  (Gray property)"

**Camera**: Stable 3/4 view.

**Overlay title**: "⑧ Gray Map  φ: Z₄ → {0,1}²"
**Subtitle**: "φ(0)=00  φ(1)=01  φ(2)=11  φ(3)=10"
**Note**: "Hamming distance ≤ Lee distance — enables binary search over fingerprints"

---

### ⑨ Fingerprint (frames 505–569)

**Visual**
- Rubik's-cube style encoding: all six cube faces rendered with 3×3 grids of
  ABCD-coloured cells, fading in (α: 0.15 → 0.95).
- Cube rotates slowly.
- After 55% progress, text appears:
    `6 faces × 9 cells × 2 bits = 108 bits`
    `Compact to 64-bit via Lee-distance projection`

**Camera**: Slow rotation, elev≈25°.

**Overlay title**: "⑨ Q² Fingerprint — Lee-Metric Surface Encoding"
**Subtitle**: "All embedding information projected onto 6 faces of the unit cube"
**Note**: "Colour = quaternary region {A=0, B=1, C=2, D=3} → 64-bit fingerprint"

---

## Geometry Notes for Implementors

### Octahedron vertex and face ordering

Vertices (L1 unit ball, `‖v‖₁ = 1`):
```
top  = (0, 0, +1)   bot = (0, 0, −1)
eq   = [(+1,0,0), (0,+1,0), (−1,0,0), (0,−1,0)]   ← ordered round equator
```

Upper faces (apex = top):  face k = [top, eq[k], eq[(k+1)%4]]  → color QUAT[k]
Lower faces (apex = bot):   face k = [bot, eq[k], eq[(k+1)%4]]  → color QUAT[(3−k)%4]

**Note**: eq must go *around* the equator in order, not in ±x ±y order.
The original code used `[s,0,0],[-s,0,0],[0,s,0],[0,-s,0]` which pairs opposite
vertices and creates degenerate flat faces. The correct order is above.

### Sphere → Octahedron morph formula

For each point `P` on the unit L2 sphere:
```
P_octa = P / ‖P‖₁
P(t)   = (1−t)·P + t·P_octa
```
This continuously deforms the sphere to the octahedron.
Works for both the word-point scatter and the full wireframe mesh.

### Complement involution colour lerp

```python
import matplotlib.colors as mc

def lerp_color(c1, c2, t):
    r1,g1,b1 = mc.to_rgb(c1)
    r2,g2,b2 = mc.to_rgb(c2)
    return (r1+(r2-r1)*t, g1+(g2-g1)*t, b1+(b2-b1)*t)

# pulse ∈ [0,1], oscillating
comp_colors = [
    lerp_color(CYAN,    LIME,    pulse),   # A → C
    lerp_color(MAGENTA, ORANGE,  pulse),   # B → D
    lerp_color(LIME,    CYAN,    pulse),   # C → A
    lerp_color(ORANGE,  MAGENTA, pulse),   # D → B
]
```
