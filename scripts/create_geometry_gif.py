import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ====================== SETUP ======================
fig = plt.figure(figsize=(14, 9), facecolor='#0a0a1f')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0a0a1f')
fig.patch.set_facecolor('#0a0a1f')
ax.set_axis_off()

# Neon palette (A=cyan, B=magenta, C=lime, D=orange)
neon = ['#00f0ff', '#ff00aa', '#aaff00', '#ffaa00']

# Data for each phase
np.random.seed(42)
n = 800
theta = np.random.uniform(0, np.pi, n)
phi = np.random.uniform(0, 2*np.pi, n)
x_sph = np.sin(theta) * np.cos(phi)
y_sph = np.sin(theta) * np.sin(phi)
z_sph = np.cos(theta)

# Cross-polytope (octahedron)
octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
x_octa = np.repeat(octa[:,0], n//6 + 1)[:n] * 1.1
y_octa = np.repeat(octa[:,1], n//6 + 1)[:n] * 1.1
z_octa = np.repeat(octa[:,2], n//6 + 1)[:n] * 1.1

# Cube vertices for final phase
cube_verts = np.array(np.meshgrid([-1,1],[-1,1],[-1,1])).T.reshape(-1,3)
x_cube = np.repeat(cube_verts[:,0], n//8 + 1)[:n] * 0.95
y_cube = np.repeat(cube_verts[:,1], n//8 + 1)[:n] * 0.95
z_cube = np.repeat(cube_verts[:,2], n//8 + 1)[:n] * 0.95

# Quaternary groups (for coloring)
groups = np.random.randint(0, 4, n)

title = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha='center', color='white', fontsize=18, fontweight='bold')
sub = ax.text2D(0.5, 0.88, "", transform=ax.transAxes, ha='center', color='#00f0ff', fontsize=12)
scatter = ax.scatter([], [], [], s=8, alpha=0.9)

phases = [60, 80, 70, 60, 50]  # frame counts per stage
total = sum(phases)

def get_phase(f):
    if f < phases[0]: return 0, f/phases[0]
    if f < sum(phases[:2]): return 1, (f-phases[0])/phases[1]
    if f < sum(phases[:3]): return 2, (f-sum(phases[:2]))/phases[2]
    if f < sum(phases[:4]): return 3, (f-sum(phases[:3]))/phases[3]
    return 4, (f-sum(phases[:4]))/phases[4]

def update(f):
    ax.cla()
    ax.set_axis_off()
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_zlim(-1.3,1.3)
    phase, prog = get_phase(f)

    if phase == 0:  # 1. Hypersphere
        x,y,z = x_sph, y_sph, z_sph
        c = '#00f0ff'
        title.set_text("1. Unit Hypersphere – L2 Embeddings")
        sub.set_text("cosine similarity space • concentration of measure")
    elif phase == 1:  # 2. Cross-Polytope
        t = prog
        x = (1-t)*x_sph + t*x_octa
        y = (1-t)*y_sph + t*y_octa
        z = (1-t)*z_sph + t*z_octa
        c = '#cc00ff'
        title.set_text("2. Cross-Polytope (L1 Geometry)")
        sub.set_text("coordinate decoupling • surface-area information bound")
    elif phase == 2:  # 3. Quaternary Quantization
        x,y,z = x_sph, y_sph, z_sph
        c = [neon[g] for g in groups]
        title.set_text("3. Quaternary Quantization (A/B/C/D)")
        sub.set_text("Gray coding → Lee distance = Hamming distance")
    elif phase == 3:  # 4. Gray Hypercube
        t = prog
        x = (1-t)*x_sph + t*x_cube
        y = (1-t)*y_sph + t*y_cube
        z = (1-t)*z_sph + t*z_cube
        c = '#aaff00'
        title.set_text("4. Gray-coded Binary Hypercube")
        sub.set_text("run-reduction → transition sequence packing")
    else:  # 5. Solvable Cube / 64-bit Key
        x,y,z = x_cube*1.05, y_cube*1.05, z_cube*1.05
        c = '#ff00aa'
        title.set_text("5. 64-bit Solvable Key")
        sub.set_text("Lee-metric searchable semantic fingerprint")

    ax.scatter(x, y, z, c=c, s=9 if phase==2 else 7, alpha=0.85, edgecolors='white', linewidth=0.3)
    ax.view_init(elev=25, azim=f*0.8)   # smooth orbit

    return []

ani = FuncAnimation(fig, update, frames=total, interval=40, blit=False)

print("Rendering q2_geometry_evolution.gif ... (≈25 seconds)")
ani.save('q2_geometry_evolution.gif', writer='pillow', fps=18, dpi=120)
print("✅ GIF saved! Open q2_geometry_evolution.gif")
plt.close()
