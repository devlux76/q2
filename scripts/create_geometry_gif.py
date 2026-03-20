"""
Q² Geometry Animation
=====================
9-phase narrative following ANIMATION_SCRIPT.md:

  ① seed            – words scatter in the unit cube
  ② l2_project      – project radially onto L2 sphere
  ③ pi4_declare     – π = 4 declaration
  ④ sphere_to_octa  – sphere mesh morphs to octahedron  ← KEY VISUAL
  ⑤ z4_label        – ABCD Z4 labels and Lee-metric sequence
  ⑥ grid_lines      – quadrant grid lines
  ⑦ complement      – complement involution A↔C  B↔D
  ⑧ gray_code       – Gray map φ: Z4 → {0,1}²
  ⑨ fingerprint     – Rubik's cube Lee-metric fingerprint

Run:  python scripts/create_geometry_gif.py
Out:  q2_geometry_evolution.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── palette ───────────────────────────────────────────────────────────────────
BG      = '#0a0a1f'
WHITE   = '#ffffff'
CYAN    = '#00f0ff'   # A = 0
MAGENTA = '#ff00aa'   # B = 1
LIME    = '#aaff00'   # C = 2
ORANGE  = '#ffaa00'   # D = 3
BLUE    = '#3355ff'
PURPLE  = '#9933ff'

QUAT = [CYAN, MAGENTA, LIME, ORANGE]   # indexed by Z4 value

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), facecolor=BG)
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# ── math helpers ──────────────────────────────────────────────────────────────

def smooth(t: float) -> float:
    """Cubic ease-in-out."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def l2_norm_rows(pts: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def l1_norm_rows(pts: np.ndarray) -> np.ndarray:
    n = np.sum(np.abs(pts), axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def lerp_color(c1: str, c2: str, t: float) -> tuple:
    r1, g1, b1 = mc.to_rgb(c1)
    r2, g2, b2 = mc.to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


# ── phase schedule ────────────────────────────────────────────────────────────
PHASES = [
    ('seed',           50),
    ('l2_project',     75),
    ('pi4_declare',    60),
    ('sphere_to_octa', 80),
    ('z4_label',       65),
    ('grid_lines',     65),
    ('complement',     55),
    ('gray_code',      55),
    ('fingerprint',    65),
]
_starts = np.cumsum([0] + [d for _, d in PHASES])
TOTAL   = int(_starts[-1])   # 570


def get_phase(f: int):
    for k, (name, dur) in enumerate(PHASES):
        if f < _starts[k + 1]:
            return k, name, (f - _starts[k]) / dur
    return len(PHASES) - 1, PHASES[-1][0], 1.0


# ── word data ─────────────────────────────────────────────────────────────────
PHRASE  = "I can believe 10 impossible things before breakfast"
WORDS   = PHRASE.split()
N       = len(WORDS)
WCOLORS = ([CYAN, MAGENTA, LIME, ORANGE, BLUE, PURPLE] * 3)[:N]

rng = np.random.default_rng(42)
# scatter inside unit ball
_dirs   = rng.normal(size=(N, 3))
_dirs   = l2_norm_rows(_dirs)
_radii  = rng.random(N) ** (1 / 3)
scatter_pts = _dirs * _radii[:, None]

sphere_pts  = l2_norm_rows(scatter_pts)   # on L2 unit sphere
octa_pts    = l1_norm_rows(sphere_pts)    # on L1 unit sphere (octahedron)


# ── draw primitives ───────────────────────────────────────────────────────────

def _set_axes():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)


def draw_cube_edges(size=1.0, color=WHITE, alpha=0.3, lw=1.2):
    s = size
    C = [(-s,-s,-s), (s,-s,-s), (s,s,-s), (-s,s,-s),
         (-s,-s, s), (s,-s, s), (s,s, s), (-s,s, s)]
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in E:
        ax.plot([C[i][0], C[j][0]], [C[i][1], C[j][1]], [C[i][2], C[j][2]],
                color=color, alpha=alpha, linewidth=lw)


def draw_sphere_wire(alpha=0.15):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 11)
    U, V = np.meshgrid(u, v)
    ax.plot_wireframe(np.cos(U) * np.sin(V),
                      np.sin(U) * np.sin(V),
                      np.cos(V),
                      color=CYAN, alpha=alpha, linewidth=0.4, rstride=2, cstride=2)


def draw_words(pts: np.ndarray, alpha=1.0, s=40):
    for i, (p, w) in enumerate(zip(pts, WORDS)):
        ax.scatter([p[0]], [p[1]], [p[2]], color=WCOLORS[i], s=s, alpha=alpha, zorder=5)
        ax.text(p[0], p[1], p[2], ' ' + w, color=WHITE, fontsize=7,
                alpha=alpha * 0.9, zorder=5)


def draw_octahedron(face_alpha=0.18, edge_alpha=0.75, lw=1.8,
                    quat_override=None):
    """
    Octahedron with vertices at ±e_i.

    Equatorial vertices go *around* the equator so adjacent faces share an edge:
        eq = [(+1,0,0), (0,+1,0), (−1,0,0), (0,−1,0)]

    Upper faces: [top, eq[k], eq[(k+1)%4]]  → QUAT[k]
    Lower faces: [bot, eq[k], eq[(k+1)%4]]  → QUAT[(3−k)%4]
    """
    colors = list(quat_override) if quat_override is not None else QUAT
    s = 1.0
    top = [0.0, 0.0,  s]
    bot = [0.0, 0.0, -s]
    # correct order: around the equator
    eq  = [[s, 0, 0], [0, s, 0], [-s, 0, 0], [0, -s, 0]]

    faces, face_colors = [], []
    for k in range(4):
        a, b = eq[k], eq[(k + 1) % 4]
        faces.append([top, a, b])
        face_colors.append(colors[k])
    for k in range(4):
        a, b = eq[k], eq[(k + 1) % 4]
        faces.append([bot, a, b])
        face_colors.append(colors[(3 - k) % 4])

    poly = Poly3DCollection(faces, alpha=face_alpha, zsort='average')
    poly.set_facecolors(face_colors)
    poly.set_edgecolor('white')
    poly.set_linewidth(lw * 0.5)
    ax.add_collection3d(poly)

    # edges
    verts = [[s, 0, 0], [0, s, 0], [-s, 0, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (0,4),(1,4),(2,4),(3,4),
             (0,5),(1,5),(2,5),(3,5)]
    for i, j in edges:
        ax.plot([verts[i][0], verts[j][0]],
                [verts[i][1], verts[j][1]],
                [verts[i][2], verts[j][2]],
                color=CYAN, alpha=edge_alpha, linewidth=lw)


def draw_grid_lines(alpha=0.6, lw=1.2):
    s = 1.0
    kw  = dict(color='#4466ff', alpha=alpha, linewidth=lw, linestyle='--')
    kw2 = dict(color='#4466ff', alpha=min(1.0, alpha * 1.4), linewidth=lw * 1.1)
    ax.plot([ 0,  0], [-s,  s], [ s,  s], **kw)
    ax.plot([-s,  s], [ 0,  0], [ s,  s], **kw)
    ax.plot([ 0,  0], [-s,  s], [-s, -s], **kw)
    ax.plot([-s,  s], [ 0,  0], [-s, -s], **kw)
    ax.plot([ 0,  0], [ 0,  0], [-s,  s], **kw2)
    ax.plot([ 0,  0], [-s,  s], [ 0,  0], **kw2)
    ax.plot([-s,  s], [ 0,  0], [ 0,  0], **kw2)


def label_abcd(alpha=1.0, size=13):
    kw = dict(fontsize=size, fontweight='bold', ha='center', va='center', alpha=alpha)
    ax.text( 0.65,  0.65, 1.15, 'A\n(0)', color=CYAN,    **kw)
    ax.text(-0.65,  0.65, 1.15, 'B\n(1)', color=MAGENTA, **kw)
    ax.text(-0.65, -0.65, 1.15, 'C\n(2)', color=LIME,    **kw)
    ax.text( 0.65, -0.65, 1.15, 'D\n(3)', color=ORANGE,  **kw)


def overlay(title: str, subtitle: str, note: str = '', progress: float = 0.0):
    ax.text2D(0.5, 0.97, title,
              transform=ax.transAxes, ha='center', va='top',
              color=WHITE, fontsize=13, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3f',
                        alpha=0.75, edgecolor=CYAN))
    ax.text2D(0.5, 0.89, subtitle,
              transform=ax.transAxes, ha='center', va='top',
              color=CYAN, fontsize=9, style='italic')
    if note:
        ax.text2D(0.5, 0.04, note,
                  transform=ax.transAxes, ha='center', va='bottom',
                  color='#9999cc', fontsize=8)
    filled = int(progress * 36)
    bar    = '█' * filled + '░' * (36 - filled)
    ax.text2D(0.02, 0.01, bar,
              transform=ax.transAxes, ha='left', va='bottom',
              color=CYAN, fontsize=6.5, family='monospace')
    ax.text2D(0.96, 0.01, f'{int(progress * 100)}%',
              transform=ax.transAxes, ha='right', va='bottom',
              color=CYAN, fontsize=8)


# ── Rubik's face colours ──────────────────────────────────────────────────────
_rng2      = np.random.default_rng(99)
FACE_NAMES = ['top', 'bottom', 'front', 'back', 'right', 'left']
RUBIKS_COLORS = {}
for _fn in FACE_NAMES:
    _base  = _rng2.choice(QUAT)
    _cells = [_base] * 5 + [_rng2.choice(QUAT) for _ in range(4)]
    _rng2.shuffle(_cells)
    RUBIKS_COLORS[_fn] = list(_cells)


def draw_rubiks_face(face='top', size=1.0, colors=None, alpha=0.85):
    s = size
    if colors is None:
        colors = [CYAN] * 9
    d   = 2 * s / 3
    idx = 0
    for i in range(3):
        for j in range(3):
            u0, u1 = -s + i * d, -s + i * d + d
            v0, v1 = -s + j * d, -s + j * d + d
            if face == 'top':
                verts = [[u0, v0, s], [u1, v0, s], [u1, v1, s], [u0, v1, s]]
            elif face == 'bottom':
                verts = [[u0, v0, -s], [u1, v0, -s], [u1, v1, -s], [u0, v1, -s]]
            elif face == 'front':
                verts = [[u0, s, v0], [u1, s, v0], [u1, s, v1], [u0, s, v1]]
            elif face == 'back':
                verts = [[u0, -s, v0], [u1, -s, v0], [u1, -s, v1], [u0, -s, v1]]
            elif face == 'right':
                verts = [[s, u0, v0], [s, u1, v0], [s, u1, v1], [s, u0, v1]]
            else:  # left
                verts = [[-s, u0, v0], [-s, u1, v0], [-s, u1, v1], [-s, u0, v1]]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(colors[idx])
            poly.set_edgecolor('#000000')
            poly.set_linewidth(1.0)
            ax.add_collection3d(poly)
            idx += 1


# ── main update ───────────────────────────────────────────────────────────────

def update(f: int):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    _set_axes()

    phase_idx, phase, prog = get_phase(f)
    sp      = smooth(prog)
    overall = f / TOTAL

    # camera
    azim = 35 + f * 0.28
    if phase_idx == 0:
        elev = 70 - sp * 40
    elif phase_idx <= 3:
        elev = 30 + sp * 4
    else:
        elev = 22 + min(phase_idx - 4, 4) * 2.5
    ax.view_init(elev=float(elev), azim=float(azim))

    # ── ① seed ────────────────────────────────────────────────────────────────
    if phase == 'seed':
        n_vis = max(1, int(sp * N * 1.4))
        draw_words(scatter_pts[:n_vis], alpha=min(1.0, sp * 2.5))
        draw_cube_edges(alpha=0.12 + 0.15 * sp)
        overlay('① Text Seed',
                f'"{PHRASE}"',
                'Raw token positions in high-dimensional embedding space',
                overall)

    # ── ② l2_project ─────────────────────────────────────────────────────────
    elif phase == 'l2_project':
        draw_sphere_wire(alpha=0.08 + 0.18 * sp)
        draw_cube_edges(alpha=0.20)
        cur = scatter_pts + (sphere_pts - scatter_pts) * sp
        draw_words(cur, alpha=0.95)
        overlay('② L2 Normalise → Unit Sphere',
                r'$\|x\|_2 = 1$  ·  Words project radially onto $S^{n-1}$',
                'Problem: O(n) rotational freedom ⟹ inter-model incommensurability',
                overall)

    # ── ③ pi4_declare ────────────────────────────────────────────────────────
    elif phase == 'pi4_declare':
        draw_sphere_wire(alpha=0.15)
        draw_cube_edges(alpha=0.22)
        ax.text2D(0.5, 0.61, 'π = 4',
                  transform=ax.transAxes, ha='center', va='center',
                  color=LIME, fontsize=52, fontweight='bold', alpha=sp)
        ax.text2D(0.5, 0.46,
                  'Switch from L² to L¹ — the one impossible belief\nthat dissolves the problem',
                  transform=ax.transAxes, ha='center', va='center',
                  color=ORANGE, fontsize=10, style='italic',
                  alpha=smooth(max(0.0, sp - 0.3) / 0.7))
        overlay('③ The π = 4 Insight',
                'Hypersphere incommensurability is unsolvable in L² — so leave L²',
                'L¹ cross-polytope has rational geometry and exact π = 4',
                overall)

    # ── ④ sphere_to_octa ─────────────────────────────────────────────────────
    elif phase == 'sphere_to_octa':
        # morph sphere wireframe to octahedron surface
        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 10)
        U, V = np.meshgrid(u, v)
        sx = np.cos(U) * np.sin(V)
        sy = np.sin(U) * np.sin(V)
        sz = np.cos(V)
        l1n = np.abs(sx) + np.abs(sy) + np.abs(sz)
        l1n[l1n == 0] = 1.0
        # blend each mesh point toward its L1 projection
        mx = (1.0 - sp) * sx + sp * (sx / l1n)
        my = (1.0 - sp) * sy + sp * (sy / l1n)
        mz = (1.0 - sp) * sz + sp * (sz / l1n)
        ax.plot_wireframe(mx, my, mz,
                          color=CYAN, alpha=0.22, linewidth=0.5,
                          rstride=1, cstride=1)
        # morph word points
        cur = sphere_pts + (octa_pts - sphere_pts) * sp
        draw_words(cur, alpha=0.90)
        # octahedron faces solidify in the second half
        if sp > 0.45:
            draw_octahedron(face_alpha=(sp - 0.45) * 0.38,
                            edge_alpha=sp * 0.75,
                            lw=1.6)
        overlay('④ Sphere → Cross-Polytope (Octahedron)',
                r'$\|x\|_2=1 \;\longrightarrow\; \|x\|_1=1$',
                'L¹ unit ball = octahedron — 8 flat triangular faces, 6 axis vertices, ρ = 4',
                overall)

    # ── ⑤ z4_label ───────────────────────────────────────────────────────────
    elif phase == 'z4_label':
        draw_octahedron(face_alpha=0.22, edge_alpha=0.85, lw=1.9)
        draw_words(octa_pts, alpha=0.90)
        if sp > 0.35:
            label_abcd(alpha=smooth((sp - 0.35) / 0.65))
        if sp > 0.60:
            a = smooth((sp - 0.60) / 0.40)
            ax.text2D(0.5, 0.44,
                      'A(0) → B(1) → C(2) → D(3) → A(0)   [mod 4]',
                      transform=ax.transAxes, ha='center', va='center',
                      color=LIME, fontsize=10, fontweight='bold', alpha=a)
            ax.text2D(0.5, 0.37,
                      'effective ρ = 4  ·  Lee metric distance',
                      transform=ax.transAxes, ha='center', va='center',
                      color=ORANGE, fontsize=9, alpha=a)
        overlay('⑤ Z₄ Quantisation on Octahedron',
                r'$\mathbb{Z}/4\mathbb{Z}$ : four regions, Lee metric $d_L(a,b)=\min(|a-b|,\,4-|a-b|)$',
                'Each face carries one quaternary symbol: A=0  B=1  C=2  D=3',
                overall)

    # ── ⑥ grid_lines ─────────────────────────────────────────────────────────
    elif phase == 'grid_lines':
        draw_cube_edges(alpha=0.25 + 0.15 * sp, lw=1.3)
        draw_grid_lines(alpha=0.25 + 0.50 * sp)
        draw_octahedron(face_alpha=0.14, edge_alpha=0.50, lw=1.3)
        draw_words(octa_pts, alpha=0.85)
        if sp > 0.45:
            label_abcd(alpha=smooth((sp - 0.45) / 0.55), size=11)
        overlay('⑥ Grid Quadrant Lines',
                'Four dividers partition the cube into {A, B, C, D} quantisation regions',
                'Each word point lands in exactly one quaternary region',
                overall)

    # ── ⑦ complement ─────────────────────────────────────────────────────────
    elif phase == 'complement':
        # sinusoidal pulse drives the colour swap
        pulse = 0.5 + 0.5 * np.sin(prog * np.pi * 5)
        comp = [
            lerp_color(CYAN,    LIME,    pulse),   # A → C
            lerp_color(MAGENTA, ORANGE,  pulse),   # B → D
            lerp_color(LIME,    CYAN,    pulse),   # C → A
            lerp_color(ORANGE,  MAGENTA, pulse),   # D → B
        ]
        draw_octahedron(face_alpha=0.28, edge_alpha=0.88, lw=2.0,
                        quat_override=comp)
        ax.text2D(0.5, 0.46, 'θ(x) = x + 2  (mod 4)',
                  transform=ax.transAxes, ha='center', va='center',
                  color=WHITE, fontsize=14, fontweight='bold', alpha=sp)
        ax.text2D(0.5, 0.39, 'A ↔ C        B ↔ D',
                  transform=ax.transAxes, ha='center', va='center',
                  color=LIME, fontsize=12, alpha=smooth(max(0.0, sp - 0.2) / 0.8))
        overlay('⑦ Complement Involution',
                r'$\theta : x \mapsto x + 2 \;(\mathrm{mod}\; 4)$ — antipodal face swap',
                'Self-inverse: θ(θ(x)) = x  ·  maps each symbol to its antipodal partner',
                overall)

    # ── ⑧ gray_code ──────────────────────────────────────────────────────────
    elif phase == 'gray_code':
        draw_octahedron(face_alpha=0.16, edge_alpha=0.60, lw=1.5)
        if sp > 0.20:
            a = smooth((sp - 0.20) / 0.80)
            kw = dict(transform=ax.transAxes, ha='center', va='center',
                      fontsize=11, fontweight='bold', fontfamily='monospace')
            ax.text2D(0.73, 0.77, 'A = 00', color=CYAN,    alpha=a, **kw)
            ax.text2D(0.27, 0.77, 'B = 01', color=MAGENTA, alpha=a, **kw)
            ax.text2D(0.27, 0.54, 'C = 11', color=LIME,    alpha=a, **kw)
            ax.text2D(0.73, 0.54, 'D = 10', color=ORANGE,  alpha=a, **kw)
        if sp > 0.50:
            a2 = smooth((sp - 0.50) / 0.50)
            ax.text2D(0.5, 0.43,
                      '00 → 01 → 11 → 10 → 00',
                      transform=ax.transAxes, ha='center', va='center',
                      color=WHITE, fontsize=12, fontfamily='monospace', alpha=a2)
            ax.text2D(0.5, 0.36,
                      'Adjacent codes differ by exactly 1 bit  (Gray property)',
                      transform=ax.transAxes, ha='center', va='center',
                      color='#9999cc', fontsize=8.5, alpha=a2)
        overlay('⑧ Gray Map  φ: Z₄ → {0,1}²',
                'φ(0)=00  φ(1)=01  φ(2)=11  φ(3)=10',
                'Hamming distance ≤ Lee distance — enables binary search over fingerprints',
                overall)

    # ── ⑨ fingerprint ────────────────────────────────────────────────────────
    elif phase == 'fingerprint':
        for fn in ['top', 'bottom', 'front', 'back', 'right', 'left']:
            draw_rubiks_face(fn, size=1.0, colors=RUBIKS_COLORS[fn],
                             alpha=0.15 + 0.80 * sp)
        draw_cube_edges(color='#000000', alpha=0.90, lw=2.0)
        if sp > 0.55:
            a = smooth((sp - 0.55) / 0.45)
            ax.text2D(0.5, 0.44,
                      '6 faces × 9 cells × 2 bits = 108 bits',
                      transform=ax.transAxes, ha='center', va='center',
                      color=WHITE, fontsize=11, fontweight='bold', alpha=a)
            ax.text2D(0.5, 0.37,
                      'Compact to 64-bit via Lee-distance projection',
                      transform=ax.transAxes, ha='center', va='center',
                      color=CYAN, fontsize=9, alpha=a)
        overlay('⑨ Q² Fingerprint — Lee-Metric Surface Encoding',
                'All embedding information projected onto 6 faces of the unit cube',
                'Colour = quaternary region {A=0, B=1, C=2, D=3} → 64-bit fingerprint',
                overall)


# ── render ────────────────────────────────────────────────────────────────────
ani = FuncAnimation(fig, update, frames=TOTAL, interval=50, blit=False)
print(f"Rendering q2_geometry_evolution.gif  ({TOTAL} frames, 15 fps) …")
ani.save('q2_geometry_evolution.gif', writer='pillow', fps=15, dpi=90)
print("✅  Saved → q2_geometry_evolution.gif")
plt.close()
