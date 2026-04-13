# ==========================================
# --- Environment Setup and Dependencies ---
# ==========================================
import sys
import subprocess

# ==========================================
# --- User-Defined Simulation Parameters ---
# ==========================================

# --- Physics & Engine Selection ---
PHYSICS_ENGINE = 'pychrono' # Options: 'pymunk', 'pychrono'
                          # 'pymunk'    — fast 2-D rigid-body physics (strictly 2D).
                          # 'pychrono'  — high-accuracy 3-D NSC + HHT physics (Project Chrono).
                          #               install: conda install -c projectchrono pychrono

# --- Geometry, Materials, and Initial State ---
GRID_SIZE_MM = 1000.0           # XY dimension of the containing box (mm)
BLOCK_SIZE_M = 0.05             # Base size for blocks (m)
BLOCK_DENSITY = 1000.0          # kg/m^3
FRICTION = 0.6                  # Friction coefficient for all surfaces
INITIAL_SLOPE_DEG = 30.0        # Initial slope of blocks leaning against the right wall
SPHERE_PERCENTAGE = 50          # Percentage of blocks that are spheres (0 to 100)
B_AXIS_MULTIPLIER = 3           # For non-sphere blocks, multiplier for the b-axis relative to a-axis

# --- PyChrono 3D-Specific Parameters ---
# These are ignored if PHYSICS_ENGINE is 'pymunk'.
PYCHRONO_GRID_WIDTH_M = BLOCK_SIZE_M * 5.0 # Width of the grid in the 3rd dimension (Z-axis)
PYCHRONO_Z_DISTRIBUTION = 'random' # How to place blocks in the 3rd dimension (Z-axis).
                                   # 'central': All blocks placed in a single layer at Z=0.
                                   # 'uniform': Blocks placed in a regular, staggered 3D grid.
                                   # 'random':  Blocks placed at random Z-positions.

# --- Tilt Timing ---
PAUSE_BEFORE_TILT_S = 5.0       # Initial pause before tilting begins (s)
TILT_DURATION_S = 30.0          # Duration of the tilt from 0 to MAX_TILT_DEG (s)
MAX_TILT_DEG = 180.0            # Maximum angle the box will tilt to

# --- Export Options ---
EXPORT_NETCDF = False
NETCDF_FILE   = "slopefailure.nc"

# --- Derived Parameters (do not modify) ---
L = GRID_SIZE_MM / 1000.0  # 1.0 m

# Automatically check and install required Python dependencies
def install_dependencies():
    required_packages = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "IPython": "ipython",
    }

    if PHYSICS_ENGINE == 'pymunk':
        required_packages["pymunk"] = "pymunk"
    # pychrono CANNOT be installed via pip — the pip package (pychrono 1.x) is unrelated.
    # The real Project Chrono bindings must be installed via conda (see comment at top).
    # We skip auto-install here and validate the correct package is present at import time.
    if EXPORT_NETCDF and PHYSICS_ENGINE == 'pychrono':
        required_packages["netCDF4"] = "netCDF4"

    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Missing '{module}'. Installing '{package}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"\n[!] ERROR: Failed to install '{package}' via pip.")
                if package == "pychrono":
                    print("    PyChrono is best installed via conda (pip may not work):")
                    print("    conda install -c projectchrono pychrono")
                    print("    See: https://projectchrono.org/")
                    print("    Alternatively, set PHYSICS_ENGINE = 'pymunk' at the top of this script.\n")
                sys.exit(1)

install_dependencies()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation
from IPython.display import HTML, Video, display
import math

# ==========================================
# --- Physics Engine Implementations ---
# ==========================================

if PHYSICS_ENGINE == 'pymunk':
    import pymunk
    def create_near_square_pymunk(space, pos, width, height, density):
        # Keep perturb minor so blocks stack realistically
        perturb = min(width, height) * 0.02
        
        vertices = [
            (-width/2 + np.random.uniform(-perturb, perturb), -height/2 + np.random.uniform(-perturb, perturb)),
            (width/2 + np.random.uniform(-perturb, perturb), -height/2 + np.random.uniform(-perturb, perturb)),
            (width/2 + np.random.uniform(-perturb, perturb), height/2 + np.random.uniform(-perturb, perturb)),
            (-width/2 + np.random.uniform(-perturb, perturb), height/2 + np.random.uniform(-perturb, perturb))
        ]

        mass = density * (width * height * width)
        moment = pymunk.moment_for_poly(mass, vertices)

        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = np.random.uniform(-0.1, 0.1)

        shape = pymunk.Poly(body, vertices)
        shape.friction = FRICTION
        shape.elasticity = 0.1

        space.add(body, shape)
        return shape

    def create_sphere_approx_pymunk(space, pos, radius, density):
        num_sides = 16
        vertices = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            vertices.append((radius * math.cos(angle), radius * math.sin(angle)))

        mass = density * (math.pi * radius * radius * radius * 2)
        moment = pymunk.moment_for_poly(mass, vertices)

        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = np.random.uniform(-0.2, 0.2)

        shape = pymunk.Poly(body, vertices)
        shape.friction = FRICTION
        shape.elasticity = 0.1

        space.add(body, shape)
        return shape

    class _PMBlock:
        def __init__(self, shape, is_sphere=False, r=None, he=None):
            self.shape = shape
            self.body = shape.body
            self.is_sphere = is_sphere
            self._r = r
            self._he = he
        def get_local_vertices(self):
            return [(v.x, v.y) for v in self.shape.get_vertices()]

    def setup_simulation_pymunk():
        space = pymunk.Space()
        space.gravity = (0, -9.81)

        static_body = space.static_body
        floor = pymunk.Segment(static_body, (0, 0), (L, 0), 0.02)
        right_wall = pymunk.Segment(static_body, (L, 0), (L, L), 0.02)
        left_wall = pymunk.Segment(static_body, (0, 0), (0, L), 0.02)
        ceiling = pymunk.Segment(static_body, (0, L), (L, L), 0.02)

        for wall in [floor, right_wall, left_wall, ceiling]:
            wall.friction = FRICTION
            space.add(wall)

        blocks = []
        slope_rad = math.radians(INITIAL_SLOPE_DEG)
        x_start = L - (L * 0.8)

        # Dense Staggered (Brick-Wall) Packing
        dx = BLOCK_SIZE_M * 1.02
        dy = BLOCK_SIZE_M * B_AXIS_MULTIPLIER * 1.02
        
        y = dy / 2.0
        row = 0
        while y <= (L - x_start) * math.tan(slope_rad):
            x_offset = (dx * 0.5) if row % 2 == 1 else 0.0
            x = x_start + x_offset
            while x < L - BLOCK_SIZE_M / 2.0:
                max_height = (x - x_start) * math.tan(slope_rad)
                if y <= max_height:
                    xp = x + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                    yp = y + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                    if np.random.uniform(0, 100) < SPHERE_PERCENTAGE:
                        r_sp = BLOCK_SIZE_M / 2.0
                        shape = create_sphere_approx_pymunk(space, (xp, yp), r_sp, BLOCK_DENSITY)
                        blocks.append(_PMBlock(shape, is_sphere=True, r=r_sp))
                    else:
                        width = BLOCK_SIZE_M + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                        height = BLOCK_SIZE_M * B_AXIS_MULTIPLIER + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                        shape = create_near_square_pymunk(space, (xp, yp), width, height, BLOCK_DENSITY)
                        blocks.append(_PMBlock(shape, is_sphere=False, he=(width/2, height/2)))
                x += dx
            y += dy
            row += 1

        print(f"Pre-settling {len(blocks)} blocks in 2-D (pymunk)...")
        for i in range(300):
            space.step(1/50.0)
            if i % 50 == 49:
                print(f"  ... settle step {i+1}/300")

        return space, blocks

elif PHYSICS_ENGINE == 'pychrono':
    try:
        import pychrono as chrono
    except ImportError:
        print("\n[!] ERROR: 'pychrono' is not installed.")
        print("    The real Project Chrono bindings cannot be installed via pip.")
        print("    Install via conda with SSL verification disabled:")
        print("      conda config --set ssl_verify false")
        print("      conda install -c conda-forge -c projectchrono pychrono")
        print("    Or set PHYSICS_ENGINE = 'pymunk' at the top of this script.\n")
        sys.exit(1)

    # Guard against the fake 'pychrono' package on PyPI (v1.x) which shares the name
    # but has nothing to do with Project Chrono.
    if not hasattr(chrono, 'ChSystemNSC'):
        print("\n[!] ERROR: The installed 'pychrono' package is NOT the real Project Chrono.")
        print("    You have the unrelated PyPI package (pychrono ~1.x) instead.")
        print("    Fix:")
        print("      1. Uninstall the fake package:  pip uninstall pychrono -y")
        print("      2. Install the real one via conda:")
        print("           conda config --set ssl_verify false")
        print("           conda install -c conda-forge -c projectchrono pychrono")
        print("    Or set PHYSICS_ENGINE = 'pymunk' at the top of this script.\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # PyChrono API version shims — supports both v4-6 ("old") and v7+.
    # The key difference is ChVectorD → ChVector3d and other renames.
    # ------------------------------------------------------------------
    _CN = hasattr(chrono, 'ChVector3d')   # True = new API (v7+)

    def _v3(x, y, z=0.0):
        return chrono.ChVector3d(x, y, z) if _CN else chrono.ChVectorD(x, y, z)

    def _set_grav(sys, gx, gy):
        v = _v3(gx, gy)
        (sys.SetGravitationalAcceleration if _CN else sys.Set_G_acc)(v)

    def _get_grav(sys):
        g = (sys.GetGravitationalAcceleration() if _CN else sys.Get_G_acc())
        return (g.x, g.y)

    def _nsc_mat():
        cls = chrono.ChContactMaterialNSC if _CN else chrono.ChMaterialSurfaceNSC
        m = cls()
        m.SetFriction(FRICTION)
        m.SetRestitution(0.1)
        return m

    def _set_fixed(b, v):
        (b.SetFixed if _CN else b.SetBodyFixed)(v)

    def _qz(angle):
        """Quaternion from rotation angle around Z axis."""
        if hasattr(chrono, 'Q_from_AngZ'):
            return chrono.Q_from_AngZ(angle)
        q = chrono.ChQuaterniond()
        q.SetFromAngleZ(angle)
        return q

    def _angle_z(body):
        """Return the rotation angle around Z from a quaternion — works in all builds."""
        q = body.GetRot()
        # q = (e0, e1, e2, e3) = (w, x, y, z) in Chrono convention
        # angle_z = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        try:
            w, x, y, z = q.e0, q.e1, q.e2, q.e3
        except AttributeError:
            # newer API uses .w .x .y .z
            w, x, y, z = q.w, q.x, q.y, q.z
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _easy_box(mat, sx, sy, sz, density, fixed=False):
        b = None
        try:
            # Chrono v9+: material as 5th argument
            b = chrono.ChBodyEasyBox(sx, sy, sz, density, mat)
        except TypeError:
            try:
                # Chrono v6-8: material as 7th argument
                b = chrono.ChBodyEasyBox(sx, sy, sz, density, True, True, mat)
            except TypeError:
                # Ultimate fallback
                b = chrono.ChBodyEasyBox(sx, sy, sz, density, True, True)
                _assigned = False
                try:
                    cs = chrono.ChCollisionShapeBox(mat, sx / 2, sy / 2, sz / 2)
                    b.GetCollisionModel().ClearModel()
                    b.AddCollisionShape(cs)
                    b.GetCollisionModel().BuildModel()
                    _assigned = True
                except Exception:
                    pass
                if not _assigned:
                    for _attr in ('GetMaterialSurface', 'GetMaterialSurfaceNSC'):
                        try:
                            s = getattr(b, _attr)()
                            s.SetFriction(FRICTION)
                            s.SetRestitution(0.1)
                            _assigned = True
                            break
                        except Exception:
                            pass
                if not _assigned:
                    print("[WARNING] Could not assign contact material to box — blocks may fall through floor")
        _set_fixed(b, fixed)
        return b

    def _easy_sphere(mat, r, density):
        b = None
        try:
            # Chrono v9+: material as 3rd argument
            b = chrono.ChBodyEasySphere(r, density, mat)
        except TypeError:
            try:
                # Chrono v6-8
                b = chrono.ChBodyEasySphere(r, density, True, True, mat)
            except TypeError:
                b = chrono.ChBodyEasySphere(r, density, True, True)
                _assigned = False
                try:
                    cs = chrono.ChCollisionShapeSphere(mat, r)
                    b.GetCollisionModel().ClearModel()
                    b.AddCollisionShape(cs)
                    b.GetCollisionModel().BuildModel()
                    _assigned = True
                except Exception:
                    pass
                if not _assigned:
                    for _attr in ('GetMaterialSurface', 'GetMaterialSurfaceNSC'):
                        try:
                            s = getattr(b, _attr)()
                            s.SetFriction(FRICTION)
                            s.SetRestitution(0.1)
                            _assigned = True
                            break
                        except Exception:
                            pass
                if not _assigned:
                    print("[WARNING] Could not assign contact material to sphere — spheres may fall through floor")
        return b

    # ------------------------------------------------------------------
    # Unified-interface wrappers  —  match the pymunk / box2d block API
    # used by the animation loop (body.position.x/y, body.angle,
    # block.get_vertices()).
    # ------------------------------------------------------------------
    class _V2:
        __slots__ = ('x', 'y')
        def __init__(self, x, y): self.x = x; self.y = y

    class _CBody:
        def __init__(self, b): self._b = b
        @property
        def position(self): p = self._b.GetPos(); return _V2(p.x, p.y)
        @property
        def angle(self): return _angle_z(self._b)

    class _CBlock:
        def __init__(self, b, he=None, r=None):
            self.body = _CBody(b)
            self._he = he   # half-extents (hx, hy)  for boxes
            self._r  = r    # radius                  for spheres
            self.is_sphere = (r is not None)
        def get_local_vertices(self):
            if self._he:
                hx, hy = self._he
                return [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
            r, n = self._r, 16
            return [(r * math.cos(2 * math.pi * i / n),
                     r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    class _CSpace:
        def __init__(self, sys): self._s = sys
        @property
        def gravity(self): return _get_grav(self._s)
        @gravity.setter
        def gravity(self, val): _set_grav(self._s, val[0], val[1])
        def step(self, dt):
            # Try the step; if the integrator hits minimum step size, retry
            # with progressively smaller substeps before giving up.
            n = 1
            while n <= 16:
                try:
                    sub_dt = dt / n
                    for _ in range(n):
                        self._s.DoStepDynamics(sub_dt)
                    return
                except RuntimeError:
                    n *= 2
            # If still failing, just skip this frame (better than a crash)
            pass

    # ------------------------------------------------------------------
    def setup_simulation_pychrono():
        """
        Full 3-D rigid-body simulation using Project Chrono NSC.

        The domain is L × L in XY (the cross-section shown in the plot)
        and W_3D deep in Z.  W_3D is set to 5 block-diameters — wide
        enough that end-wall friction does not meaningfully affect the
        XY slope dynamics.  Blocks are seeded at random Z positions
        across the full width so the packing is genuinely 3-D.

        The 2-D animation plots the XY projection (ignoring Z), which
        is equivalent to a central cross-section view.
        """
        sys = chrono.ChSystemNSC()
        try:
            sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        except AttributeError:
            pass  # Fallback for older PyChrono versions that default to Bullet
        _set_grav(sys, 0.0, -9.81)

        # ── Solver ────────────────────────────────────────────────────
        try:
            slvr = chrono.ChSolverSOR()
            slvr.SetMaxIterations(200)
            slvr.SetOmega(0.8)
            sys.SetSolver(slvr)
        except AttributeError:
            try:
                sys.SetMaxItersSolverSpeed(200)
                sys.SetMaxItersSolverStab(200)
            except Exception:
                pass

        # ── Timestepper: fixed-step Euler (avoids adaptive min-step) ──
        for _type_name in ('Type_EULER_IMPLICIT_LINEARIZED',
                           'Type_EULER_IMPLICIT',
                           'Type_EULER_EXPLICIT'):
            try:
                _t = getattr(chrono.ChTimestepper, _type_name, None)
                if _t is not None:
                    sys.SetTimestepperType(_t)
                    break
            except Exception:
                continue

        mat = _nsc_mat()
        wt  = 0.02                       # wall half-thickness

        # 3-D depth from user parameters
        W_3D = PYCHRONO_GRID_WIDTH_M     # full Z width of the domain
        HW   = W_3D / 2.0                # half-width (domain centred on z=0)

        # ── Boundary walls ─────────────────────────────────────────────
        wall_specs = [
            # (cx,       cy,      cz,       sx,          sy,          sz      )
            # Floor  y = 0
            (L/2,     -wt,     0.0,      L+4*wt,      wt*2,        W_3D+4*wt),
            # Ceiling y = L
            (L/2,     L+wt,    0.0,      L+4*wt,      wt*2,        W_3D+4*wt),
            # Right wall  x = L
            (L+wt,    L/2,     0.0,      wt*2,        L+4*wt,      W_3D+4*wt),
            # Left wall   x = 0
            (-wt,     L/2,     0.0,      wt*2,        L+4*wt,      W_3D+4*wt),
            # Front wall  z = +HW
            (L/2,     L/2,     HW+wt,    L+4*wt,      L+4*wt,      wt*2    ),
            # Back wall   z = -HW
            (L/2,     L/2,    -HW-wt,    L+4*wt,      L+4*wt,      wt*2    ),
        ]
        for cx_w, cy_w, cz_w, sx, sy, sz in wall_specs:
            b = _easy_box(mat, sx, sy, sz, 1.0, fixed=True)
            b.SetPos(_v3(cx_w, cy_w, cz_w))
            sys.Add(b)

        # ── Block initialisation (full 3-D packing) ───────────────────
        blocks  = []
        slope_r = math.radians(INITIAL_SLOPE_DEG)
        x_start = L - L * 0.8
        # Block Z-size equals block XY size (cubic blocks / spheres)
        BZ      = BLOCK_SIZE_M / 2.0    # half-extent in Z
        
        # Dense Staggered (Brick-Wall) 3D Packing
        dx = BLOCK_SIZE_M * 1.02
        dy = BLOCK_SIZE_M * B_AXIS_MULTIPLIER * 1.02
        dz = BLOCK_SIZE_M * 1.02

        y = dy / 2.0
        row = 0
        while y <= (L - x_start) * math.tan(slope_r):
            x_offset = (dx * 0.5) if row % 2 == 1 else 0.0
            z_offset = (dz * 0.5) if row % 2 == 1 else 0.0

            x = x_start + x_offset
            while x < L - BLOCK_SIZE_M / 2.0:
                max_h = (x - x_start) * math.tan(slope_r)
                if y <= max_h:
                    # --- Z-Positioning Logic ---
                    # Determine the z-positions for this (x,y) column based on user setting
                    z_grid_points = []
                    if PYCHRONO_Z_DISTRIBUTION == 'central':
                        # Place all blocks in a single layer at z=0
                        z_grid_points = [0.0]
                    else: # 'uniform' or 'random'
                        # For both uniform and random, we iterate through the same grid
                        # points to maintain a similar number of total blocks.
                        z_iter = -HW + BZ + wt + z_offset
                        while z_iter < HW - BZ - wt:
                            z_grid_points.append(z_iter)
                            z_iter += dz

                    for z_base in z_grid_points:
                        xp = x + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                        yp = y + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)

                        if PYCHRONO_Z_DISTRIBUTION == 'random':
                            zp = np.random.uniform(-HW + BZ, HW - BZ)
                        elif PYCHRONO_Z_DISTRIBUTION == 'uniform':
                            zp = z_base + np.random.uniform(-BLOCK_SIZE_M*0.02, BLOCK_SIZE_M*0.02)
                        else: # 'central'
                            zp = z_base # which is 0.0

                        if np.random.uniform(0, 100) < SPHERE_PERCENTAGE:
                            r_sp = BLOCK_SIZE_M / 2.0
                            b = _easy_sphere(mat, r_sp, BLOCK_DENSITY)
                            b.SetPos(_v3(xp, yp, zp))
                            sys.Add(b)
                            blocks.append(_CBlock(b, r=r_sp))
                        else:
                            perturb = BLOCK_SIZE_M * 0.02
                            hx = BLOCK_SIZE_M / 2 + np.random.uniform(-perturb, perturb)
                            hy = (BLOCK_SIZE_M * B_AXIS_MULTIPLIER) / 2 + np.random.uniform(-perturb, perturb)
                            b = _easy_box(mat, hx * 2, hy * 2, BLOCK_SIZE_M, BLOCK_DENSITY)
                            b.SetPos(_v3(xp, yp, zp))
                            b.SetRot(_qz(np.random.uniform(-0.1, 0.1)))
                            sys.Add(b)
                            blocks.append(_CBlock(b, he=(hx, hy)))
                x += dx
            y += dy
            row += 1

        print(f"Pre-settling {len(blocks)} blocks in 3-D (pychrono)...")
        for i in range(400):
            sys.DoStepDynamics(1.0 / 100.0)
            if i % 100 == 99:
                print(f"  ... settle step {i+1}/400")

        if blocks:
            p = blocks[0].body.position
            print(f"  Done. First block pos: x={p.x:.3f}  y={p.y:.3f}")

        return _CSpace(sys), blocks

def setup_simulation():
    if PHYSICS_ENGINE == 'pymunk':
        return setup_simulation_pymunk()
    elif PHYSICS_ENGINE == 'pychrono':
        return setup_simulation_pychrono()
    else:
        raise ValueError(f"Unknown physics engine '{PHYSICS_ENGINE}'. Use 'pymunk' or 'pychrono'.")

# ==========================================
# --- Animation and Visualization Setup ---
# ==========================================

# Initialize the simulation
space, blocks = setup_simulation()

# --- Diagnostic: print positions and save a static PNG ---
if blocks:
    _bx = np.array([b.body.position.x for b in blocks])
    _by = np.array([b.body.position.y for b in blocks])
    print(f"\n[Diagnostic] {len(blocks)} blocks after settling.")
    print(f"  X: {_bx.min():.3f} \u2013 {_bx.max():.3f} m  (domain 0\u2013{L:.1f})")
    print(f"  Y: {_by.min():.3f} \u2013 {_by.max():.3f} m  (floor at 0)")
    _figdbg, _axdbg = plt.subplots(figsize=(7, 5))
    _cdbg = ['steelblue' if b.is_sphere else 'saddlebrown' for b in blocks]
    _axdbg.scatter(_bx, _by, c=_cdbg, s=15, alpha=0.85, zorder=3)
    _axdbg.add_patch(plt.Rectangle((0, 0), L, L, fill=False, ec='black', lw=2))
    _axdbg.set_xlim(-0.05, L + 0.05); _axdbg.set_ylim(-0.05, L + 0.05)
    _axdbg.set_xlabel('X (m)'); _axdbg.set_ylabel('Y height (m)')
    _axdbg.set_title(f'Initial block positions \u2013 {PHYSICS_ENGINE} ({len(blocks)} blocks)')
    _axdbg.set_aspect('equal'); _axdbg.grid(True, alpha=0.3)
    _figdbg.tight_layout()
    _figdbg.savefig('blocks_initial.png', dpi=120)
    plt.close(_figdbg)
    print("  Saved blocks_initial.png \u2014 open this to verify block positions before the video runs.")
else:
    print("[WARNING] No blocks were created!")

# --- Figure: 2D simulation view (XY cross-section + angle chart) -----
cx, cy = L / 2.0, L / 2.0
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 9),
                               gridspec_kw={'height_ratios': [4, 1]})
ax.set_xlim(cx - L * 0.85, cx + L * 0.85)
ax.set_ylim(cy - L * 0.85, cy + L * 0.85)
ax.set_aspect('equal')
ax.set_xlabel('Width (m)'); ax.set_ylabel('Height (m)')
ax.grid(True, alpha=0.2)

# Create polygon patches to faithfully render the exact dimensions and rotations
patch_artists = []
for b in blocks:
    c = 'steelblue' if b.is_sphere else 'saddlebrown'
    # Initialize with dummy coordinates
    poly = Polygon([[0,0], [0,1], [1,1], [1,0]], closed=True, 
                   facecolor=c, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=5)
    ax.add_patch(poly)
    patch_artists.append(poly)

# Domain boundary + surface fit
box_line,     = ax.plot([], [], 'k-', lw=2.5, zorder=4)
surface_line, = ax.plot([], [], 'r--', lw=2,  zorder=6, label='Surface fit')
ax.legend(loc='upper left', fontsize=8)

# --- Angle tracking subplot ---
gravity_magnitude = 9.81
FPS = 30.0
TOTAL_TIME_S  = PAUSE_BEFORE_TILT_S + TILT_DURATION_S + 5.0
TOTAL_FRAMES  = int(TOTAL_TIME_S * FPS)

ax2.set_xlim(0, TOTAL_TIME_S); ax2.set_ylim(0, 90)
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Angle (\u00b0)')
max_line, = ax2.plot([], [], 'b-', lw=2,   label='Max')
med_line, = ax2.plot([], [], 'g-', lw=2,   label='Median')
cur_line, = ax2.plot([], [], 'r-', lw=1.5, label='Current')
ax2.legend(loc='upper left'); ax2.grid(True)

recorded_surface_angles = []
time_history = []; max_angle_history = []; median_angle_history = []; current_angle_history = []

# --- NetCDF accumulators ---
_nc_t, _nc_x, _nc_y, _nc_z, _nc_a = [], [], [], [], []
if EXPORT_NETCDF and PHYSICS_ENGINE == 'pychrono':
    _nc_type = np.array([1 if b.is_sphere else 0 for b in blocks], dtype=np.int8)
    _nc_size = np.array([b._r if b.is_sphere else max(b._he) for b in blocks])

def init():
    recorded_surface_angles.clear()
    time_history.clear(); max_angle_history.clear()
    median_angle_history.clear(); current_angle_history.clear()
    box_line.set_data([], []); surface_line.set_data([], [])
    max_line.set_data([], []); med_line.set_data([], []); cur_line.set_data([], [])
    return [box_line, surface_line, max_line, med_line, cur_line] + patch_artists

def animate(frame):
    t = frame / FPS

    if t <= PAUSE_BEFORE_TILT_S:
        tilt_angle_deg = 0.0
    elif t <= PAUSE_BEFORE_TILT_S + TILT_DURATION_S:
        tilt_angle_deg = ((t - PAUSE_BEFORE_TILT_S) / TILT_DURATION_S) * MAX_TILT_DEG
    else:
        tilt_angle_deg = MAX_TILT_DEG

    tilt_rad = math.radians(tilt_angle_deg)
    space.gravity = (-gravity_magnitude * math.sin(tilt_rad),
                     -gravity_magnitude * math.cos(tilt_rad))
    for _ in range(5):
        space.step(1.0 / FPS / 5)

    # Collect XY positions (both engines expose the same .body.position interface)
    sim_x = np.array([b.body.position.x for b in blocks])
    sim_y = np.array([b.body.position.y for b in blocks])
    if PHYSICS_ENGINE == 'pychrono':
        sim_z = np.array([float(b.body._b.GetPos().z) for b in blocks])
    else:
        sim_z = np.zeros(len(blocks))

    # Diagnostics for first few frames — confirms if blocks are in the right place
    if frame in (0, 10, 30, 60) and len(blocks) > 0:
        _vy = sim_y[np.isfinite(sim_y)]
        print(f"[Frame {frame:4d} | t={t:.2f}s]  "
              f"y_min={_vy.min():.3f}  y_max={_vy.max():.3f}  y_mean={_vy.mean():.3f}")
    if frame == 0:
        n_fin = int(np.sum(np.isfinite(sim_x) & np.isfinite(sim_y)))
        print(f"  {n_fin}/{len(blocks)} blocks have finite positions.")
        for i in range(min(3, len(blocks))):
            print(f"  block[{i}]: x={sim_x[i]:.3f}  y={sim_y[i]:.3f}  z={sim_z[i]:.3f}")

    # Rotate the domain (and blocks) to visualise tilt
    cos_t = math.cos(tilt_rad); sin_t = math.sin(tilt_rad)
    def transform(x, y):
        rx = cx + (x - cx) * cos_t - (y - cy) * sin_t
        ry = cy + (x - cx) * sin_t + (y - cy) * cos_t
        return rx, ry

    # Update domain box wireframe
    bx_pts = [0, 0, L, L, 0]; by_pts = [L, 0, 0, L, L]
    tx_pts, ty_pts = zip(*[transform(x, y) for x, y in zip(bx_pts, by_pts)])
    box_line.set_data(tx_pts, ty_pts)

    # Update exact polygon positions and rotations
    for b, poly in zip(blocks, patch_artists):
        b_sim_x = b.body.position.x
        b_sim_y = b.body.position.y

        if not (np.isfinite(b_sim_x) and np.isfinite(b_sim_y) and b_sim_y > -BLOCK_SIZE_M * 2 and b_sim_x > -L and b_sim_x < 2 * L):
            poly.set_xy([[cx - 3 * L, cy - 3 * L]])
            continue

        angle = b.body.angle
        c_a = math.cos(angle)
        s_a = math.sin(angle)

        world_verts = []
        for lx, ly in b.get_local_vertices():
            rx = lx * c_a - ly * s_a
            ry = lx * s_a + ly * c_a
            wx = b_sim_x + rx
            wy = b_sim_y + ry
            tx, ty = transform(wx, wy)
            world_verts.append((tx, ty))

        poly.set_xy(world_verts)

    # Surface fit — only use blocks that are still inside the domain
    valid = (np.isfinite(sim_x) & np.isfinite(sim_y)
             & (sim_x >= 0) & (sim_x <= L)
             & (sim_y >= 0))
    sx_v = sim_x[valid]; sy_v = sim_y[valid]
    if np.sum(valid) > 1:
        bins = np.linspace(0, L, 21)
        bidx = np.digitize(sx_v, bins)
        top_x, top_y = [], []
        for i in range(1, 21):
            mask = bidx == i
            if np.any(mask):
                top_x.append(float(np.mean(sx_v[mask])))
                top_y.append(float(np.max(sy_v[mask])))
        if len(top_x) > 1:
            m, c_fit = np.polyfit(top_x, top_y, 1)
            lx = np.array([min(top_x), max(top_x)])
            ly = m * lx + c_fit
            tx_l, ty_l = zip(*[transform(xl, yl) for xl, yl in zip(lx, ly)])
            surface_line.set_data(tx_l, ty_l)
            slope_deg = abs(math.degrees(math.atan(m)))
            recorded_surface_angles.append(slope_deg)
            cur_max = np.max(recorded_surface_angles)
            cur_med = np.median(recorded_surface_angles)
            time_history.append(t)
            max_angle_history.append(cur_max)
            median_angle_history.append(cur_med)
            current_angle_history.append(slope_deg)
            ax.set_title(
                f"t={t:.1f}s  Tilt={tilt_angle_deg:.1f}\u00b0  "
                f"Slope={slope_deg:.1f}\u00b0  Max={cur_max:.1f}\u00b0", fontsize=9)
        else:
            surface_line.set_data([], [])
    else:
        surface_line.set_data([], [])

    max_line.set_data(time_history, max_angle_history)
    med_line.set_data(time_history, median_angle_history)
    cur_line.set_data(time_history, current_angle_history)

    if EXPORT_NETCDF and PHYSICS_ENGINE == 'pychrono':
        _nc_t.append(t)
        _nc_x.append(sim_x.tolist())
        _nc_y.append(sim_y.tolist())
        _nc_z.append(sim_z.tolist())
        _nc_a.append([b.body.angle for b in blocks])

    return [box_line, surface_line, max_line, med_line, cur_line] + patch_artists

print(f"Generating simulation video ({TOTAL_FRAMES} frames at {FPS} fps)...")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=TOTAL_FRAMES, interval=1000/FPS, blit=False)

writer = animation.FFMpegWriter(fps=FPS, metadata=dict(title='Slope Failure Simulation'), bitrate=1800)
anim.save("slopefailure.mp4", writer=writer)
plt.close()
print("Successfully saved slopefailure.mp4!")

# --- Save NetCDF for ParaView -----------------------------------------
if EXPORT_NETCDF and PHYSICS_ENGINE == 'pychrono' and _nc_t:
    import netCDF4 as nc4
    with nc4.Dataset(NETCDF_FILE, 'w', format='NETCDF4') as ds:
        ds.createDimension('time',  len(_nc_t))
        ds.createDimension('block', len(blocks))
        vt  = ds.createVariable('time',       'f4', ('time',))
        vx  = ds.createVariable('x',          'f4', ('time', 'block'))
        vy  = ds.createVariable('y',          'f4', ('time', 'block'))
        vz  = ds.createVariable('z',          'f4', ('time', 'block'))
        va  = ds.createVariable('angle_z',    'f4', ('time', 'block'))
        vbt = ds.createVariable('block_type', 'i1', ('block',))
        vsz = ds.createVariable('size',       'f4', ('block',))
        vt.units = 'seconds'
        vx.units = vy.units = vz.units = 'meters'
        va.units = 'radians'
        vbt.long_name = '0=box  1=sphere'
        vsz.long_name = 'half-extent (box) or radius (sphere)  [m]'
        vt[:]  = np.array(_nc_t,  dtype=np.float32)
        vx[:]  = np.array(_nc_x,  dtype=np.float32)
        vy[:]  = np.array(_nc_y,  dtype=np.float32)
        vz[:]  = np.array(_nc_z,  dtype=np.float32)
        va[:]  = np.array(_nc_a,  dtype=np.float32)
        vbt[:] = _nc_type
        vsz[:] = _nc_size.astype(np.float32)
    print(f"Saved NetCDF: {NETCDF_FILE}")
    print("ParaView: File>Open>.nc  →  TableToPoints(x/y/z)  →  Glyph(size)  →  Play")

# ==========================================
# --- Results Output ---
# ==========================================
if recorded_surface_angles:
    print(f"Maximum Global Surface Slope:  {np.max(recorded_surface_angles):.1f}°")
    print(f"Median Global Surface Slope:   {np.median(recorded_surface_angles):.1f}°")

display(Video("slopefailure.mp4", embed=True, html_attributes="controls loop autoplay"))
