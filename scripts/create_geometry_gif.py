"""
Q2 Embedding Geometry – Animated Visual Guide
==============================================
Redesigned following reviewer feedback:
  • Every transition has explanatory text
  • Starts with a 3D cube and 4 grid-lines (top-down)
  • Unit ball shown as two square-base pyramids stacked base-to-base (octahedron)
  • Camera pulls from top-down → side-on to reveal prism structure
  • Pyramids stack to fill cube cells
  • Final frame: Rubik's-cube projection with quaternary colours on each face

Run:  python scripts/create_geometry_gif.py
Output: q2_geometry_evolution.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ───────────────────────── palette ─────────────────────────
BG     = '#0a0a1f'
WHITE  = '#ffffff'
CYAN   = '#00f0ff'
MAGENTA= '#ff00aa'
LIME   = '#aaff00'
ORANGE = '#ffaa00'
BLUE   = '#3355ff'
PURPLE = '#9933ff'

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
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        ax.plot([corners[i][0],corners[j][0]],
                [corners[i][1],corners[j][1]],
                [corners[i][2],corners[j][2]],
                color=color, alpha=alpha, linewidth=lw)


def draw_grid_lines(size=1.0, color='#4466ff', alpha=0.6, lw=1.2):
    """4 internal grid lines that create the A/B/C/D quadrants."""
    s = size
    kw = dict(color=color, alpha=alpha, linewidth=lw, linestyle='--')
    # Two lines on the TOP face dividing into four quadrants
    ax.plot([ 0, 0], [-s, s], [s, s], **kw)   # front-back mid line
    ax.plot([-s, s], [ 0, 0], [s, s], **kw)   # left-right mid line
    # Same two lines on BOTTOM face
    ax.plot([ 0, 0], [-s, s], [-s,-s], **kw)
    ax.plot([-s, s], [ 0, 0], [-s,-s], **kw)
    # Vertical mid-plane lines (interior axes)
    kw2 = dict(color=color, alpha=min(1.0, alpha*1.4), linewidth=lw*1.1)
    ax.plot([ 0, 0], [ 0, 0], [-s, s], **kw2)   # vertical axis
    ax.plot([ 0, 0], [-s, s], [ 0, 0], **kw2)   # depth axis
    ax.plot([-s, s], [ 0, 0], [ 0, 0], **kw2)   # horizontal axis


def _octa_verts_faces(size=1.0, z_offset=0.0):
    s = size
    top  = np.array([0, 0,  s + z_offset])
    bot  = np.array([0, 0, -s + z_offset])
    equator = np.array([[s,0,z_offset],[-s,0,z_offset],
                        [0,s,z_offset],[0,-s,z_offset]])
    faces = []
    colors = []
    # Upper pyramid (4 triangular faces)
    fc_top = [CYAN, MAGENTA, LIME, ORANGE]
    for k in range(4):
        a = equator[k];  b = equator[(k+1) % 4]
        faces.append([top.tolist(), a.tolist(), b.tolist()])
        colors.append(fc_top[k])
    # Lower pyramid (4 triangular faces)
    fc_bot = [ORANGE, LIME, MAGENTA, CYAN]
    for k in range(4):
        a = equator[k];  b = equator[(k+1) % 4]
        faces.append([bot.tolist(), a.tolist(), b.tolist()])
        colors.append(fc_bot[k])
    return faces, colors


def draw_octahedron(size=1.0, face_alpha=0.18, edge_alpha=0.75,
                    edge_lw=1.8, z_offset=0.0):
    faces, colors = _octa_verts_faces(size, z_offset)
    poly = Poly3DCollection(faces, alpha=face_alpha, zsort='average')
    poly.set_facecolors(colors)
    poly.set_edgecolor('white')
    poly.set_linewidth(edge_lw * 0.5)
    ax.add_collection3d(poly)
    # Bold edge lines
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
    """8 corner prisms – the complement of the octahedron inside the cube."""
    s = 1.0
    # Each octant: one triangular prism from the equatorial square to a cube corner
    octants = [(1,1,1),(-1,1,1),(-1,-1,1),(1,-1,1),
               (1,1,-1),(-1,1,-1),(-1,-1,-1),(1,-1,-1)]
    clrs = [CYAN, MAGENTA, LIME, ORANGE,
            ORANGE, LIME, MAGENTA, CYAN]
    for (sx,sy,sz), clr in zip(octants, clrs):
        # Octahedron vertex in this octant (on equator if z≠0, else on z-axis)
        apex  = [0, 0, sz * s]               # top/bottom octahedron tip
        e1    = [sx*s, 0,  0]               # equatorial vertex on x axis
        e2    = [0,  sy*s, 0]               # equatorial vertex on y axis
        corner= [sx*s, sy*s, sz*s]          # cube corner
        # Prism faces
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
            u0 = -s + i*d;  u1 = u0 + d
            v0 = -s + j*d;  v1 = v0 + d
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
            else:  # left
                verts = [[-s,u0,v0],[-s,u1,v0],[-s,u1,v1],[-s,u0,v1]]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(colors[idx])
            poly.set_edgecolor('#000000')
            poly.set_linewidth(1.2)
            ax.add_collection3d(poly)
            idx += 1


def label_quadrants(alpha=1.0):
    """ABCD labels on the top face."""
    kw = dict(fontsize=15, fontweight='bold', ha='center', va='center', alpha=alpha)
    ax.text( 0.5, 0.5, 1.08, 'A', color=CYAN,    **kw)
    ax.text(-0.5, 0.5, 1.08, 'B', color=MAGENTA, **kw)
    ax.text(-0.5,-0.5, 1.08, 'C', color=LIME,    **kw)
    ax.text( 0.5,-0.5, 1.08, 'D', color=ORANGE,  **kw)


def add_overlay(title_str, subtitle_str, note_str='', progress=0.0):
    """Consistent text overlay for every frame."""
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
    # progress bar
    filled = int(progress * 36)
    bar = '█' * filled + '░' * (36 - filled)
    ax.text2D(0.02, 0.01, bar,
              transform=ax.transAxes, ha='left', va='bottom',
              color=CYAN, fontsize=6.5, family='monospace')
    ax.text2D(0.96, 0.01, f'{int(progress*100)}%',
              transform=ax.transAxes, ha='right', va='bottom',
              color=CYAN, fontsize=8)


# ───────────────────── phase schedule ──────────────────────
# (name, n_frames)
PHASES = [
    ('text_intro',     70),   # intro phrase
    ('sphere_project', 80),   # project words from interior to sphere surface
    ('hypersphere',    80),   # evolve into a hyper-sphere concept
    ('wave',           70),   # wave sampling + incommensurability math
    ('p4_hypercube',   80),   # p=4 boundary hypercube
    ('curiouser',      50),   # curiouser and curiouser
    ('begin_end',      60),   # begin at the beginning ... end
    ('grid',           65),   # top-down, draw 3D cube + 4 grid lines + ABCD labels
    ('unit_ball',      65),   # add octahedron (two pyramids base-to-base)
    ('rotate',         75),   # camera: top-down → side-on
    ('prisms',         65),   # reveal prism decomposition
    ('stack',          65),   # top pyramid lifts, bottom drops → cubes stack
    ('rubiks',         65),   # Rubik's cube projection finale
]
_starts = np.cumsum([0] + [d for _, d in PHASES])
TOTAL   = int(_starts[-1])


def get_phase(f: int):
    for k, (name, dur) in enumerate(PHASES):
        if f < _starts[k+1]:
            return k, name, (f - _starts[k]) / dur
    return len(PHASES)-1, PHASES[-1][0], 1.0


# ─────────────────────── rubiks colours ────────────────────
# 6 faces × 9 cells – each cell gets a quaternary colour
rng = np.random.Generator(np.random.PCG64(42))
FACE_NAMES = ['top','bottom','front','back','right','left']
RUBIKS_COLORS = {}
for fn in FACE_NAMES:
    base = rng.choice(QUAT)
    cells = [base]*5  # centre + 4 adjacent stay same colour
    cells += [rng.choice(QUAT) for _ in range(4)]
    rng.shuffle(cells)
    RUBIKS_COLORS[fn] = cells


# ──────────────────────── update ───────────────────────────

def update(f: int):
    ax.cla()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_zlim(-1.45, 1.45)

    phase_idx, phase_name, prog = get_phase(f)
    sp  = smooth(prog)
    overall = f / TOTAL

    # ── camera angles ────────────────────────────────────────
    AZIM_BASE = 35 + f * 0.4   # slow orbit throughout
    if phase_idx <= 1:          # top-down
        elev = 90.0 - sp * 8   # almost bird-eye, tiny tilt for depth
    elif phase_idx == 2:        # rotate down
        elev = 82.0 - sp * 62  # 82° → 20°
    else:                       # side-on, slight rise per phase
        elev = 20.0 + (phase_idx - 3) * 3

    ax.view_init(elev=elev, azim=AZIM_BASE)

    # ─── PHASE 0 – Grid Formation ────────────────────────────
    if phase_name == 'grid':
        draw_cube_edges(alpha=0.15 + 0.35*sp, lw=1.4)
        draw_grid_lines(alpha=0.2 + 0.6*sp, lw=1.2)
        if sp > 0.45:
            label_quadrants(alpha=smooth((sp-0.45)/0.55))
        add_overlay(
            '① Grid Formation — Embedding Space',
            'Four grid lines divide the unit cube into quaternary regions {A, B, C, D}',
            note_str='Each region captures one of the four Gray-coded coordinate states',
            progress=overall,
        )

    # ─── PHASE 1 – Unit Ball (octahedron) ────────────────────
    elif phase_name == 'unit_ball':
        draw_cube_edges(alpha=0.35, lw=1.4)
        draw_grid_lines(alpha=0.65, lw=1.2)
        label_quadrants()
        draw_octahedron(face_alpha=0.08 + 0.20*sp, edge_alpha=0.3 + 0.6*sp,
                        edge_lw=1.4 + 0.6*sp)
        # annotation arrow stub – text only (3-D arrows need extra deps)
        if sp > 0.5:
            a = smooth((sp-0.5)/0.5)
            ax.text2D(0.85, 0.72,
                      '← upper\n   pyramid',
                      transform=ax.transAxes, color=CYAN,
                      fontsize=8, ha='center', alpha=a)
            ax.text2D(0.85, 0.30,
                      '← lower\n   pyramid',
                      transform=ax.transAxes, color=MAGENTA,
                      fontsize=8, ha='center', alpha=a)
        add_overlay(
            '② Unit Ball in ℓ¹ Space — Two Pyramids Base-to-Base',
            'The cross-polytope (octahedron) is the ℓ¹ unit ball:  ‖x‖₁ ≤ 1',
            note_str='Upper pyramid (apex up) + lower pyramid (apex down) share the equatorial square',
            progress=overall,
        )

    # ─── PHASE 2 – Rotate camera top-down → side-on ──────────
    elif phase_name == 'rotate':
        draw_cube_edges(alpha=0.35, lw=1.4)
        draw_grid_lines(alpha=0.65, lw=1.2)
        label_quadrants(alpha=max(0.0, 1.0 - sp*1.5))
        draw_octahedron(face_alpha=0.25, edge_alpha=0.8, edge_lw=1.8)
        add_overlay(
            '③ Rotating View — Top-Down → Side-On',
            'Tilting the camera reveals the full 3-D geometry of the unit ball inside the cube',
            note_str='The equatorial square of the octahedron aligns with the grid cross-section',
            progress=overall,
        )

    # ─── PHASE 3 – Prisms ────────────────────────────────────
    elif phase_name == 'prisms':
        draw_cube_edges(alpha=0.30, lw=1.4)
        draw_grid_lines(alpha=0.55, lw=1.1)
        draw_octahedron(face_alpha=0.15, edge_alpha=0.5, edge_lw=1.4)
        draw_prisms(alpha=0.08 + 0.22*sp)
        add_overlay(
            '④ Grid × Unit Ball = Corner Prisms',
            'Each grid cell (octant) contains a pyramid-prism slice of the cross-polytope',
            note_str='Prism = space between the octahedron face and the cube corner — '
                     'information capacity of one Gray-coded region',
            progress=overall,
        )

    # ─── PHASE 4 – Stack pyramids to form cubes ──────────────
    elif phase_name == 'stack':
        offset = sp * 0.55           # pyramids separate
        draw_cube_edges(alpha=0.30, lw=1.4)
        draw_grid_lines(alpha=0.45, lw=1.0)
        draw_octahedron(face_alpha=0.12, edge_alpha=0.45,
                        edge_lw=1.4, z_offset=0.0)
        # Emphasise the split with a dashed equatorial ring
        theta_ = np.linspace(0, 2*np.pi, 64)
        ax.plot(np.cos(theta_), np.sin(theta_), [0]*64,
                color=WHITE, alpha=0.4, linewidth=1.0, linestyle=':')
        # Drifting labels
        if sp > 0.25:
            a = smooth((sp-0.25)/0.75)
            ax.text2D(0.5, 0.75, '↑ upper half-cube', transform=ax.transAxes,
                      ha='center', color=CYAN, fontsize=9, alpha=a)
            ax.text2D(0.5, 0.22, '↓ lower half-cube', transform=ax.transAxes,
                      ha='center', color=MAGENTA, fontsize=9, alpha=a)
        add_overlay(
            '⑤ Stacking Pyramids Fills the Cube',
            'Upper and lower pyramids separate along the equator — each half fills one cube layer',
            note_str='Stacking four pyramids base-to-base tiles the full unit hypercube without overlap',
            progress=overall,
        )

    # ─── PHASE 5 – Rubik's cube ──────────────────────────────
    else:
        for fn in FACE_NAMES:
            draw_rubiks_face(fn, size=1.0,
                             colors=RUBIKS_COLORS[fn],
                             alpha=0.12 + 0.78*sp)
        draw_cube_edges(color='#000000', alpha=0.9, lw=2.2)
        if sp > 0.6:
            a = smooth((sp-0.6)/0.4)
            for fn, (tx, ty) in [('top',(0.5,0.88)),('front',(0.76,0.58)),
                                  ('right',(0.83,0.42))]:
                ax.text2D(tx, ty, fn, transform=ax.transAxes,
                          ha='center', color=WHITE, fontsize=8, alpha=a*0.7)
        add_overlay(
            '⑥ Rubik\'s Cube — Information on Surface Faces',
            'All embedding information projected onto the 6 cube faces (9 cells each)',
            note_str='Colour of each cell = quaternary region {A=■, B=■, C=■, D=■}'
                     '  → 64-bit Lee-metric searchable fingerprint',
            progress=overall,
        )


# ───────────────────────── render ──────────────────────────
ani = FuncAnimation(fig, update, frames=TOTAL, interval=50, blit=False)

print(f"Rendering q2_geometry_evolution.gif  ({TOTAL} frames) …")
ani.save('q2_geometry_evolution.gif', writer='pillow', fps=15, dpi=110)
print("✅  GIF saved →  q2_geometry_evolution.gif")
plt.close()
