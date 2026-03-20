"""
Q2 Embedding Geometry – Animated Visual Guide
==============================================
Extended to include:
  • Phrase -> sphere projection
  • Sphere -> hypersphere evolution and incommensurability math
  • ℓ_p boundary transformation (p=2 → p=4) to hypercube style
  • 'curiouser and curiouser', 'Begin at the beginning...'
  • Existing quaternary cube/octagon section continues

Run:  python scripts/create_geometry_gif.py
Output: q2_geometry_evolution.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ───────────────────────── palette ─────────────────────────
BG      = '#0a0a1f'
WHITE   = '#ffffff'
CYAN    = '#00f0ff'
MAGENTA = '#ff00aa'
LIME    = '#aaff00'
ORANGE  = '#ffaa00'
BLUE    = '#3355ff'
PURPLE  = '#9933ff'

QUAT = [CYAN, MAGENTA, LIME, ORANGE]   # A B C D

# ───────────────────────── figure ──────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

# ───────────────────────── helpers ─────────────────────────

def smooth(t: float) -> float:
    """Cubic ease-in-out."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def draw_cube_edges(size=1.0, color=WHITE, alpha=0.35, lw=1.4):
    s = size
    corners = [(-s,-s,-s),(s,-s,-s),(s,s,-s),(-s,s,-s),
               (-s,-s, s),(s,-s, s),(s,s, s),(-s,s, s)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        ax.plot([corners[i][0],corners[j][0]],
                [corners[i][1],corners[j][1]],
                [corners[i][2],corners[j][2]],
                color=color, alpha=alpha, linewidth=lw)


def spherical_projection(p):
    # project points in ball to unit sphere by radial normalization
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return p / norm


def lp_norm(pts, p=2.0, axis=1, keepdims=True):
    return np.sum(np.abs(pts)**p, axis=axis, keepdims=keepdims)**(1.0/p)


def lp_unit_projection(pts, p=2.0):
    norm_p = lp_norm(pts, p=p, axis=1, keepdims=True)
    norm_p[norm_p == 0] = 1.0
    return pts / norm_p


def draw_text_center(message, size=24, alpha=1.0):
    ax.text2D(0.5, 0.6, message, transform=ax.transAxes,
              ha='center', va='center', color=WHITE,
              fontsize=size, fontweight='bold', alpha=alpha)


def draw_word_points(pts, words, colors, alpha=1.0, s=45):
    for i, (p, w, c) in enumerate(zip(pts, words, colors)):
        ax.scatter([p[0]], [p[1]], [p[2]], color=c, s=s, alpha=alpha)
        if i % 2 == 0:
            ax.text(p[0], p[1], p[2], w, color=WHITE, fontsize=8, alpha=alpha*0.95)
        else:
            ax.text(p[0], p[1], p[2], w, color=WHITE, fontsize=8, alpha=alpha*0.95)


# ───────────────────── phase schedule ──────────────────────
PHASES = [
    ('text_intro',     60),
    ('sphere_project', 80),
    ('pi4_impossible', 70),
    ('hypersphere',    80),
    ('wave',           70),
    ('p4_hypercube',   80),
    ('curiouser',      50),
    ('begin_end',      60),
    ('grid',           65),
    ('unit_ball',      65),
    ('rotate',         75),
    ('prisms',         65),
    ('stack',          65),
    ('rubiks',         65),
]
_starts = np.cumsum([0] + [d for _, d in PHASES])
TOTAL   = int(_starts[-1])


def get_phase(f: int):
    for k, (name, dur) in enumerate(PHASES):
        if f < _starts[k+1]:
            return k, name, (f - _starts[k]) / dur
    return len(PHASES)-1, PHASES[-1][0], 1.0


# ───────────────────── input phrase + points ─────────────────
PHRASE = "I can believe 10 impossible things before breakfast"
WORDS = PHRASE.split()
NWORDS = len(WORDS)
WORD_COLORS = [CYAN, MAGENTA, LIME, ORANGE, BLUE, PURPLE] * 5

rng = np.random.default_rng(42)
inner_radii = rng.random(NWORDS)**(1 / 3)
inner_dirs = rng.normal(size=(NWORDS, 3))
inner_dirs /= np.linalg.norm(inner_dirs, axis=1, keepdims=True)
inner_points = inner_dirs * inner_radii[:, None]

sphere_points = spherical_projection(inner_points)

# ───────────────────── grid / octahedron / prisms / rubiks helpers ─────────

def draw_grid_lines(size=1.0, color='#4466ff', alpha=0.6, lw=1.2):
    """4 internal grid lines that create the A/B/C/D quadrants."""
    s = size
    kw = dict(color=color, alpha=alpha, linewidth=lw, linestyle='--')
    ax.plot([ 0, 0], [-s, s], [s, s], **kw)
    ax.plot([-s, s], [ 0, 0], [s, s], **kw)
    ax.plot([ 0, 0], [-s, s], [-s,-s], **kw)
    ax.plot([-s, s], [ 0, 0], [-s,-s], **kw)
    kw2 = dict(color=color, alpha=min(1.0, alpha*1.4), linewidth=lw*1.1)
    ax.plot([ 0, 0], [ 0, 0], [-s, s], **kw2)
    ax.plot([ 0, 0], [-s, s], [ 0, 0], **kw2)
    ax.plot([-s, s], [ 0, 0], [ 0, 0], **kw2)


def label_quadrants(alpha=1.0):
    """ABCD labels on the top face."""
    kw = dict(fontsize=15, fontweight='bold', ha='center', va='center', alpha=alpha)
    ax.text( 0.5, 0.5, 1.08, 'A', color=CYAN,    **kw)
    ax.text(-0.5, 0.5, 1.08, 'B', color=MAGENTA, **kw)
    ax.text(-0.5,-0.5, 1.08, 'C', color=LIME,    **kw)
    ax.text( 0.5,-0.5, 1.08, 'D', color=ORANGE,  **kw)


def _octa_verts_faces(size=1.0, z_offset=0.0):
    s = size
    top  = np.array([0, 0,  s + z_offset])
    bot  = np.array([0, 0, -s + z_offset])
    equator = np.array([[s,0,z_offset],[-s,0,z_offset],
                        [0,s,z_offset],[0,-s,z_offset]])
    faces = []
    colors = []
    for k in range(4):
        a = equator[k]; b = equator[(k+1) % 4]
        faces.append([top.tolist(), a.tolist(), b.tolist()])
        colors.append(QUAT[k])
    for k in range(4):
        a = equator[k]; b = equator[(k+1) % 4]
        faces.append([bot.tolist(), a.tolist(), b.tolist()])
        colors.append(QUAT[(3-k) % 4])
    return faces, colors


def draw_octahedron(size=1.0, face_alpha=0.18, edge_alpha=0.75,
                    edge_lw=1.8, z_offset=0.0):
    faces, colors = _octa_verts_faces(size, z_offset)
    poly = Poly3DCollection(faces, alpha=face_alpha, zsort='average')
    poly.set_facecolors(colors)
    poly.set_edgecolor('white')
    poly.set_linewidth(edge_lw * 0.5)
    ax.add_collection3d(poly)
    s = size
    verts = [[s,0,z_offset],[-s,0,z_offset],
             [0,s,z_offset],[0,-s,z_offset],
             [0,0,s+z_offset],[0,0,-s+z_offset]]
    edges = [(0,2),(0,3),(1,2),(1,3),(2,4),(3,4),(2,5),(3,5),
             (0,4),(1,4),(0,5),(1,5)]
    for i,j in edges:
        ax.plot([verts[i][0],verts[j][0]],
                [verts[i][1],verts[j][1]],
                [verts[i][2],verts[j][2]],
                color=CYAN, alpha=edge_alpha, linewidth=edge_lw)


def draw_prisms(alpha=0.22):
    """8 corner prisms – complement of octahedron inside the cube."""
    s = 1.0
    octants = [(1,1,1),(-1,1,1),(-1,-1,1),(1,-1,1),
               (1,1,-1),(-1,1,-1),(-1,-1,-1),(1,-1,-1)]
    clrs = [CYAN, MAGENTA, LIME, ORANGE, ORANGE, LIME, MAGENTA, CYAN]
    for (sx,sy,sz), clr in zip(octants, clrs):
        apex   = [0, 0, sz * s]
        e1     = [sx*s, 0,  0]
        e2     = [0,  sy*s, 0]
        corner = [sx*s, sy*s, sz*s]
        prism_faces = [
            [apex, e1, corner],
            [apex, e2, corner],
            [apex, e1, e2],
            [e1,   corner, e2],
        ]
        poly = Poly3DCollection(prism_faces, alpha=alpha, zsort='average')
        poly.set_facecolor(clr)
        poly.set_edgecolor('white')
        poly.set_linewidth(0.4)
        ax.add_collection3d(poly)


def draw_rubiks_face(face='top', size=1.0, colors=None, alpha=0.85):
    """3×3 coloured patches on one face of the cube."""
    s = size
    if colors is None:
        colors = [CYAN]*9
    d = 2*s / 3
    idx = 0
    for i in range(3):
        for j in range(3):
            u0 = -s + i*d; u1 = u0 + d
            v0 = -s + j*d; v1 = v0 + d
            if face == 'top':
                verts = [[u0,v0,s],[u1,v0,s],[u1,v1,s],[u0,v1,s]]
            elif face == 'bottom':
                verts = [[u0,v0,-s],[u1,v0,-s],[u1,v1,-s],[u0,v1,-s]]
            elif face == 'front':
                verts = [[u0,s,v0],[u1,s,v0],[u1,s,v1],[u0,s,v1]]
            elif face == 'back':
                verts = [[u0,-s,v0],[u1,-s,v0],[u1,-s,v1],[u0,-s,v1]]
            elif face == 'right':
                verts = [[s,u0,v0],[s,u1,v0],[s,u1,v1],[s,u0,v1]]
            else:
                verts = [[-s,u0,v0],[-s,u1,v0],[-s,u1,v1],[-s,u0,v1]]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(colors[idx])
            poly.set_edgecolor('#000000')
            poly.set_linewidth(1.2)
            ax.add_collection3d(poly)
            idx += 1


# ─────────────────────── rubiks colours ─────────────────────
_rng2 = np.random.Generator(np.random.PCG64(42))
FACE_NAMES = ['top','bottom','front','back','right','left']
RUBIKS_COLORS = {}
for _fn in FACE_NAMES:
    _base = _rng2.choice(QUAT)
    _cells = [_base]*5
    _cells += [_rng2.choice(QUAT) for _ in range(4)]
    _rng2.shuffle(_cells)
    RUBIKS_COLORS[_fn] = _cells

# ───────────────────── overlay helper ──────────────────────

def add_overlay(title_str, subtitle_str, note_str='', progress=0.0):
    ax.text2D(0.5, 0.97, title_str,
              transform=ax.transAxes, ha='center', va='top',
              color=WHITE, fontsize=14, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a3f',
                        alpha=0.75, edgecolor=CYAN))
    ax.text2D(0.5, 0.89, subtitle_str,
              transform=ax.transAxes, ha='center', va='top',
              color=CYAN, fontsize=9.5, style='italic')
    if note_str:
        ax.text2D(0.5, 0.04, note_str,
                  transform=ax.transAxes, ha='center', va='bottom',
                  color='#9999cc', fontsize=8.5)
    filled = int(progress * 36)
    bar = '█' * filled + '░' * (36 - filled)
    ax.text2D(0.02, 0.01, bar,
              transform=ax.transAxes, ha='left', va='bottom',
              color=CYAN, fontsize=6.5, family='monospace')
    ax.text2D(0.96, 0.01, f'{int(progress*100)}%',
              transform=ax.transAxes, ha='right', va='bottom',
              color=CYAN, fontsize=8)


# ────────────────────── update ────────────────────────────

def update(f: int):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_zlim(-1.45, 1.45)

    phase_idx, phase_name, prog = get_phase(f)
    sp = smooth(prog)
    overall = f / TOTAL

    AZIM_BASE = 40 + f * 0.3
    if phase_idx <= 1:
        elev = 85.0 - sp*30.0
    elif phase_idx == 2:
        elev = 55.0 - sp*20.0
    else:
        elev = 20.0 + min(8.0, phase_idx) * 2.0

    ax.view_init(elev=elev, azim=AZIM_BASE)

    # ---------- phase 0: phrase intro
    if phase_name == 'text_intro':
        draw_text_center(f'"{PHRASE}"', size=28, alpha=1.0)
        draw_text_center('Begin by reading the semantic seed phrase', size=14, alpha=0.85)
        add_overlay('① Text Seed', 'Initial phrase: raw tokens in language space',
                    'Embedding vectorization starts from token positions', progress=overall)

    # ---------- phase 1: project into sphere
    elif phase_name == 'sphere_project':
        draw_cube_edges(alpha=0.25)
        # draw unit sphere wireframe
        _u = np.linspace(0, 2*np.pi, 32)
        _v = np.linspace(0, np.pi, 16)
        _U, _V = np.meshgrid(_u, _v)
        _SX = np.cos(_U) * np.sin(_V)
        _SY = np.sin(_U) * np.sin(_V)
        _SZ = np.cos(_V)
        ax.plot_wireframe(_SX, _SY, _SZ, color=CYAN, alpha=0.10 + 0.12*sp,
                          linewidth=0.5, rstride=2, cstride=2)

        for i, p in enumerate(inner_points):
            target = sphere_points[i]
            cur = p + (target - p) * sp
            ax.plot([p[0], cur[0], target[0]], [p[1], cur[1], target[1]], [p[2], cur[2], target[2]],
                    color='#6666aa', alpha=0.45)

        draw_word_points(inner_points + (sphere_points - inner_points) * (sp * 1.1), WORDS, WORD_COLORS, alpha=0.95)
        draw_text_center('Project words into unit sphere S^{n-1}', size=16, alpha=0.9)
        add_overlay('② Embedding Sphere',
                    r'$S^{n-1} = \{ x \in \mathbb{R}^n : \|x\|_2 = 1 \}$',
                    'Words land inside this cube — the sphere shows where they project to.\n'
                    'Problem: different models rotate the sphere freely → incommensurability',
                    progress=overall)

    # ---------- phase 2: pi=4 — the only impossible thing we need
    elif phase_name == 'pi4_impossible':
        draw_cube_edges(alpha=0.25 + 0.15*sp)
        # sphere wireframe
        _u = np.linspace(0, 2*np.pi, 32)
        _v = np.linspace(0, np.pi, 16)
        _U, _V = np.meshgrid(_u, _v)
        _SX = np.cos(_U) * np.sin(_V)
        _SY = np.sin(_U) * np.sin(_V)
        _SZ = np.cos(_V)
        ax.plot_wireframe(_SX, _SY, _SZ, color=MAGENTA, alpha=0.08 + 0.18*sp,
                          linewidth=0.5, rstride=2, cstride=2)

        # big pi=4 declaration, fades in
        ax.text2D(0.5, 0.62, 'π = 4', transform=ax.transAxes,
                  ha='center', va='center', color=LIME,
                  fontsize=48, fontweight='bold', alpha=smooth(sp))
        ax.text2D(0.5, 0.50, 'The only impossible thing we need here',
                  transform=ax.transAxes, ha='center', va='center',
                  color=ORANGE, fontsize=13, style='italic',
                  alpha=smooth(max(0.0, sp - 0.2) / 0.8))
        if sp > 0.55:
            ax.text2D(0.5, 0.40,
                      'Hypersphere incommensurability is believed unsolvable.\n'
                      'Yet — we can believe impossible things.\n'
                      'Set π = 4 and the problem becomes trivial.',
                      transform=ax.transAxes, ha='center', va='center',
                      color=WHITE, fontsize=9.5,
                      alpha=smooth((sp - 0.55) / 0.45))
        add_overlay('③ The Impossible Made Trivial',
                    r'Hypersphere rotational freedom $\Rightarrow$ incommensurability (believed unsolvable)',
                    'π = 4: the one impossible belief that dissolves the problem entirely',
                    progress=overall)

    # ---------- phase 3: hypersphere evolution
    elif phase_name == 'hypersphere':
        # represent 4th dimension by color and size variation
        w = np.sin(np.linspace(0, np.pi, NWORDS) + overall * 2.0) * 0.5 + 0.5
        x4 = sphere_points * (1.0 + w[:, None] * 0.04)
        draw_word_points(x4, WORDS, [plt.cm.viridis(cc) for cc in w], alpha=1.0, s=45)

        draw_text_center('Evolve sphere → hypersphere via extra dimension', size=16, alpha=0.9)
        add_overlay('④ Hypersphere',
                    r'$S^{n-1}$ has no preferred orientation; rotations $Q\in O(n)$ preserve semantics',
                    'Observe incommensurability: absolute axes vary across models', progress=overall)

    # ---------- phase 3: wave sampling + incommensurability
    elif phase_name == 'wave':
        tr = 1.0 + 0.12 * np.sin(overall * 4 * np.pi + np.arange(NWORDS) * 0.6)
        wave_points = sphere_points * tr[:, None]
        draw_word_points(wave_points, WORDS, WORD_COLORS, alpha=0.85, s=45)

        # draw wave surface via color mapping
        theta = np.linspace(0, 2*np.pi, 140)
        phi = np.linspace(0, np.pi, 70)
        TH, PH = np.meshgrid(theta, phi)
        R = 1.0 + 0.08 * np.sin(8*TH + overall * 5.0)
        X = R * np.sin(PH) * np.cos(TH)
        Y = R * np.sin(PH) * np.sin(TH)
        Z = R * np.cos(PH)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, color='#222277', alpha=0.14, linewidth=0)

        draw_text_center('Wave sample across embeddings (high frequency signal)', size=15, alpha=0.9)
        add_overlay('⑤ Incommensurability',
                    r'$\text{sim}(u,v)=u\cdot v,\enspace u\in S^{n-1},\enspace v\in S^{n-1}$',
                    'Different frames: without alignment, dot products vary unpredictably', progress=overall)

    # ---------- phase 4: p=4 hypercube transform
    elif phase_name == 'p4_hypercube':
        p_t = 2.0 + 2.0 * sp
        sphere_points_0 = sphere_points.copy()
        # map to lp unit boundary and blend
        lp_mapped = lp_unit_projection(sphere_points_0, p=p_t)
        # store line points
        for i in range(NWORDS):
            ax.plot([sphere_points_0[i,0], lp_mapped[i,0]],
                    [sphere_points_0[i,1], lp_mapped[i,1]],
                    [sphere_points_0[i,2], lp_mapped[i,2]],
                    color='#88ff88', alpha=0.35)

        draw_word_points(lp_mapped, WORDS, WORD_COLORS, alpha=0.95, s=40)
        draw_cube_edges(alpha=0.2 + 0.5*sp, lw=1.3)

        draw_text_center(f'ℓₚ ball morph (p={p_t:.2f}) → hypercube style', size=15, alpha=0.9)
        add_overlay('⑥ ℓₚ transition',
                    r'$\|x\|_p = (\sum_{i=1}^n |x_i|^p)^{1/p},\quad p\to 4$',
                    'p=4: boundary shape approaches hypercube facets', progress=overall)

    # ---------- phase 5: curiouser and curiouser
    elif phase_name == 'curiouser':
        draw_cube_edges(alpha=0.5)
        draw_word_points(lp_unit_projection(sphere_points, p=4.0), WORDS, WORD_COLORS, alpha=1.0, s=35)
        draw_text_center('curiouser and curiouser', size=28, alpha=sp * 0.95)
        add_overlay('⑦ Curiosity', 'The embedding journey becomes increasingly non-intuitive',
                    'This is the step where geometry feels almost magical', progress=overall)

    # ---------- phase 6: begin ... end
    elif phase_name == 'begin_end':
        draw_text_center('Begin at the beginning...', size=24, alpha=1.0 - 0.2*sp)
        draw_text_center('...and go on till you come to the end', size=18, alpha=0.2 + 0.8*sp)
        if sp > 0.6:
            draw_text_center('𐄂  The end 𐄂', size=26, alpha=(sp-0.6)/0.4)
        add_overlay('⑧ Narrative closure', 'Text closure from Lewis Carroll-inspired sequence',
                    'Stop after reaching the conceptual end', progress=overall)

    # ---------- existing phases (grid/octahedron/...) for continuation
    elif phase_name == 'grid':
        draw_cube_edges(alpha=0.15 + 0.35*sp, lw=1.4)
        draw_grid_lines(alpha=0.2 + 0.6*sp, lw=1.2)
        if sp > 0.45:
            label_quadrants(alpha=smooth((sp-0.45)/0.55))
        add_overlay(
            '⑨ Grid Formation — Embedding Space',
            'Four grid lines divide the unit cube into quaternary regions {A, B, C, D}',
            note_str='Each region captures one of the four Gray-coded coordinate states',
            progress=overall,
        )

    elif phase_name == 'unit_ball':
        draw_cube_edges(alpha=0.35, lw=1.4)
        draw_grid_lines(alpha=0.65, lw=1.2)
        label_quadrants()
        draw_octahedron(face_alpha=0.08 + 0.20*sp, edge_alpha=0.3 + 0.6*sp,
                        edge_lw=1.4 + 0.6*sp)
        if sp > 0.5:
            a = smooth((sp-0.5)/0.5)
            ax.text2D(0.85, 0.72,
                      '← upper\n   pyramid', transform=ax.transAxes,
                      color=CYAN, fontsize=8, alpha=a)
            ax.text2D(0.85, 0.30,
                      '← lower\n   pyramid', transform=ax.transAxes,
                      color=MAGENTA, fontsize=8, alpha=a)
        add_overlay(
            '⑩ Unit Ball in ℓ¹ Space — Two Pyramids Base-to-Base',
            'The cross-polytope (octahedron) is the ℓ¹ unit ball:  ‖x‖₁ ≤ 1',
            note_str='Upper pyramid (apex up) + lower pyramid (apex down) share the equatorial square',
            progress=overall,
        )

    elif phase_name == 'rotate':
        draw_cube_edges(alpha=0.35, lw=1.4)
        draw_grid_lines(alpha=0.65, lw=1.2)
        label_quadrants(alpha=max(0.0, 1.0 - sp*1.5))
        draw_octahedron(face_alpha=0.25, edge_alpha=0.8, edge_lw=1.8)
        add_overlay(
            '⑪ Rotating View — Top-Down → Side-On',
            'Tilting the camera reveals the full 3-D geometry of the unit ball inside the cube',
            note_str='The equatorial square of the octahedron aligns with the grid cross-section',
            progress=overall,
        )

    elif phase_name == 'prisms':
        draw_cube_edges(alpha=0.30, lw=1.4)
        draw_grid_lines(alpha=0.55, lw=1.1)
        draw_octahedron(face_alpha=0.15, edge_alpha=0.5, edge_lw=1.4)
        draw_prisms(alpha=0.08 + 0.22*sp)
        add_overlay(
            '⑫ Corner Prisms',
            'Each grid cell (octant) contains a pyramid-prism slice of the cross-polytope',
            note_str='Prism = space between the octahedron face and the cube corner',
            progress=overall,
        )

    elif phase_name == 'stack':
        offset = sp * 0.55
        draw_cube_edges(alpha=0.30, lw=1.4)
        draw_grid_lines(alpha=0.45, lw=1.0)
        draw_octahedron(face_alpha=0.12, edge_alpha=0.45, edge_lw=1.4, z_offset=0.0)
        theta_ = np.linspace(0, 2*np.pi, 64)
        ax.plot(np.cos(theta_), np.sin(theta_), [0]*64,
                color=WHITE, alpha=0.4, linewidth=1.0, linestyle=':')
        if sp > 0.25:
            a = smooth((sp-0.25)/0.75)
            ax.text2D(0.5, 0.75, '↑ upper half-cube', transform=ax.transAxes,
                      ha='center', color=CYAN, fontsize=9, alpha=a)
            ax.text2D(0.5, 0.22, '↓ lower half-cube', transform=ax.transAxes,
                      ha='center', color=MAGENTA, fontsize=9, alpha=a)
        add_overlay(
            '⑬ Stacking Pyramids Fills the Cube',
            'Upper and lower pyramids separate along the equator — each half fills one cube layer',
            note_str='Stacking four pyramids base-to-base tiles the full unit cube',
            progress=overall,
        )

    elif phase_name == 'rubiks':
        for fn in ['top','bottom','front','back','right','left']:
            draw_rubiks_face(fn, size=1.0, colors=RUBIKS_COLORS[fn], alpha=0.12 + 0.78*sp)
        draw_cube_edges(color='#000000', alpha=0.9, lw=2.2)
        if sp > 0.6:
            a = smooth((sp-0.6)/0.4)
            for fn, (tx, ty) in [('top',(0.5,0.88)),('front',(0.76,0.58)),('right',(0.83,0.42))]:
                ax.text2D(tx, ty, fn, transform=ax.transAxes,
                          ha='center', color=WHITE, fontsize=8, alpha=a*0.7)
        add_overlay(
            '⑭ Rubik\'s Cube — Information on Surface Faces',
            'All embedding information projected onto the 6 cube faces (9 cells each)',
            note_str='Colour = quaternary region {A, B, C, D} → 64-bit Lee-metric fingerprint',
            progress=overall,
        )


ani = FuncAnimation(fig, update, frames=TOTAL, interval=50, blit=False)

print(f"Rendering q2_geometry_evolution.gif  ({TOTAL} frames) …")
ani.save('q2_geometry_evolution.gif', writer='pillow', fps=15, dpi=110)
print("✅  GIF saved →  q2_geometry_evolution.gif")
plt.close()
