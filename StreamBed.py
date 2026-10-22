"""Prescribed-stress sediment DEM for HillslopeSim, with an adaptive outlet barrier.

Pymunk uses unit-width XZ grains; PyChrono uses real spheres/prisms in an XZ plane.
The contact solver supplies the changing pocket constraints; no Shields
threshold, prescribed grain velocity, or force-triggered teleport is used.
See StreamBed.md for the exposure/traction closure and model limitations.
"""
from __future__ import annotations

import math
import time
import numpy as np
from RiverHydraulics import RiverHydraulics, startup_factor
from DragIntegration import implicit_drag_force
from TimeStepping import computational_dt
from Recording import FrameSchedule


def validate_parameters(s):
    names = ("DOMAIN_X", "DOMAIN_Z", "SLOPE_HEIGHT", "SLOPE_LENGTH", "SLOPE_BASE",
             "PARTICLE_SIZE", "PARTICLE_DENSITY", "DT", "SETTLE_TIME", "GRAVITY_MAG",
             "FRICTION_COEFF", "TARGET_POROSITY", "INITIAL_CLEARANCE")
    names += tuple(k for k, (_, default) in s._STREAM_FIELDS.items() if not isinstance(default, str))
    if any(not math.isfinite(float(getattr(s, k))) for k in names):
        raise ValueError("Stream-bed parameters must be finite numbers.")
    for name, choices in s._STREAM_CHOICES.items():
        if getattr(s, name) not in choices:
            raise ValueError(f"{name} must be one of {choices}.")
    for name, low, high in (("FLOW_COLUMNS", 3, 256), ("FLOW_LEVELS", 16, 1024),
                             ("TURBULENCE_SEED", 0, 2**32-1)):
        value = getattr(s, name)
        if not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    if min(s.FLOW_DEPTH, s.WATER_VISCOSITY, s.ROUGHNESS_D84_FACTOR,
           s.FLOW_UPDATE_INTERVAL, s.TURBULENCE_LENGTH_X, s.TURBULENCE_LENGTH_Z,
           s.TURBULENCE_TIME_FACTOR) <= 0:
        raise ValueError("Depth, viscosity, roughness, refresh and turbulence scales must be positive.")
    if min(s.FLOW_MEAN_VELOCITY, s.STREAM_DRAG_COEFFICIENT,
           s.TURBULENCE_INTENSITY, s.TURBULENCE_STRESS_CV,
           s.STREAM_STARTUP_TIME, s.STREAM_SETTLE_DAMPING) < 0:
        raise ValueError("Velocity, drag, turbulence, startup time and settling damping must be nonnegative.")
    if not 0 < s.STREAM_CONTACT_TRAVEL_FRACTION <= .25:
        raise ValueError("Stream contact travel fraction must be in (0, 0.25].")
    if s.SIMULATE_TURBULENCE and s.TURBULENCE_TARGET == "velocity" and s.HYDRAULIC_LOAD != "drag":
        raise ValueError("Velocity turbulence requires the relative-velocity drag closure.")
    if s.STREAM_DOMAIN_SCALE <= 0:
        raise ValueError("Domain / bed scale factor must be positive and finite.")
    if s.ENGINE not in ("pymunk", "pychrono"):
        raise ValueError("Stream-bed engine must be pymunk or pychrono.")
    if not (0 <= s.SLOPE_BASE < s.SLOPE_HEIGHT < s.DOMAIN_Z - s._wall_inset()
            and 0 < s.SLOPE_LENGTH <= s.DOMAIN_X):
        raise ValueError("Stream geometry requires 0 <= base < bed top < the domain ceiling, "
                         "and 0 < bed length <= Domain X.")
    if (s.STREAM_DURATION <= 0 or s.DT <= 0 or s.SETTLE_TIME < 0 or s.GRAVITY_MAG <= 0
            or s.WATER_DENSITY <= 0 or s.PARTICLE_DENSITY <= s.WATER_DENSITY
            or s.PARTICLE_SIZE <= 0 or s.FRICTION_COEFF < 0
            or s.INITIAL_CLEARANCE < 0
            or not 0 <= s.TARGET_POROSITY < 1 or s.FRAME_INTERVAL < 1):
        raise ValueError("Use positive duration, timestep, density and grain size; sediment "
                         "density must exceed water density. Check friction and porosity.")
    if min(s.BED_STRESS_MEAN, s.BED_STRESS_AMPLITUDE, s.BED_STRESS_FREQUENCY,
           s.GRAIN_STRESS_FACTOR, s.STREAM_LIFT_RATIO) < 0:
        raise ValueError("Stress, frequency, grain stress multiplier and lift ratio must be nonnegative.")
    if s.STREAM_FORCING == "harmonic" and (s.BED_STRESS_AMPLITUDE > s.BED_STRESS_MEAN
                                          or s.BED_STRESS_FREQUENCY <= 0):
        raise ValueError("For left-to-right harmonic flow use amplitude <= mean and frequency > 0 Hz.")


def bed_stress(s, t):
    """Prescribed stress only; velocity-mode stress comes from hydraulic columns."""
    if s.STREAM_FORCING == "velocity":
        raise ValueError("Velocity forcing derives stress from u*. Read arr_stress or frames_hydraulics from the run.")
    return startup_factor(s, t)**2*(s.BED_STRESS_MEAN + (s.BED_STRESS_AMPLITUDE * math.sin(
        2 * math.pi * s.BED_STRESS_FREQUENCY * t) if s.STREAM_FORCING == "harmonic" else 0.))


def flat_particle_centres(s):
    """Nonoverlapping, staggered rows contained inside the flat target envelope."""
    rng = np.random.default_rng(s.INITIAL_SEED)
    a = s.PARTICLE_SIZE
    c = a if s.PARTICLE_SHAPE == "circle" else s._rect_c
    if c <= 0 or not math.isfinite(c):
        raise ValueError("Particle c-axis must be positive and finite.")
    random_angle = s.PARTICLE_SHAPE == "rectangle" and s.RECT_ARRANGEMENT == "mixed"
    width = height = math.hypot(a, c) if random_angle else 0.
    if not random_angle:
        width, height = a, c
    area = math.pi * a*a/4 if s.PARTICLE_SHAPE == "circle" else a*c
    spacing_factor = max(1.01, math.sqrt(area / (width * height * (1-s.TARGET_POROSITY))))
    dx, dz = width * spacing_factor, height * spacing_factor
    inset = s._wall_inset() + s.INITIAL_CLEARANCE
    result = []
    row = 0
    z = s.SLOPE_BASE + s.INITIAL_CLEARANCE + height/2
    while z + height/2 <= s.SLOPE_HEIGHT:
        x = inset + width/2 + (row % 2) * .25 * dx
        while x + width/2 <= min(s.SLOPE_LENGTH, s.DOMAIN_X-inset):
            angle = rng.uniform(0, math.pi) if random_angle else 0.
            result.append((x, z, angle, a))
            x += dx
        row += 1
        z += dz
    if not result:
        raise ValueError("No grains fit inside the stream bed. Increase bed depth/length or reduce grain size.")
    return result


def configure_chrono_stream(s, backend):
    """Constrain XZ translation; only nonspherical grains need angular locks."""
    c = backend.chrono
    widths = [size if kind == "circle" else width for kind, size, width in
              zip(backend.particle_kinds, backend.particle_sizes, backend.particle_b_sizes)]
    if widths and max(widths) > s.DOMAIN_Y*(1+1e-10):
        raise ValueError("PyChrono stream grains exceed the scaled Domain Y width. Increase Domain Y or use smaller grains.")
    backend.stream_plane_links = []
    for body, kind in zip(backend.particle_bodies, backend.particle_kinds):
        link = c.ChLinkMateGeneric()
        # A sphere's orientation does not affect its geometry. Redundant
        # angular locks introduce 1/inertia rows that badly condition global
        # gradient solvers (APGD) relative to 1/mass contact rows for small
        # grains, allowing an unconverged bed to sink under gravity.
        lock_rotation = kind != "circle"
        link.SetConstrainedCoords(False, True, False, lock_rotation, False, lock_rotation)
        link.Initialize(body, backend.wall_bodies[0], c.ChFramed(body.GetPos(), c.ChQuaterniond(1, 0, 0, 0)))
        backend.sys.AddLink(link)
        backend.stream_plane_links.append(link)


class StreamPhysics:
    """Small engine adapter: Pymunk uses unit-width loads; Chrono uses SI loads."""
    def __init__(self, s, backend):
        self.s, self.b = s, backend
        self.pymunk = hasattr(backend, "space")
        bodies = backend.particle_bodies
        self.masses = np.array([body.mass if self.pymunk else body.GetMass() for body in bodies])
        self.inertias = np.array([body.moment/s.PX_PER_M**2 if self.pymunk else body.GetInertiaXX().y for body in bodies])
        self.area_width = np.ones(len(bodies)) if self.pymunk else np.array([
            math.pi*size/4 if kind == "circle" else width
            for kind, size, width in zip(backend.particle_kinds, backend.particle_sizes, backend.particle_b_sizes)])
        self.accumulators = ([None]*len(bodies) if self.pymunk else
                             [body.AddAccumulator() if hasattr(body, "AddAccumulator") else None for body in bodies])

    def point_velocity(self, lever):
        if self.pymunk:
            spin = np.array([body.angular_velocity for body in self.b.particle_bodies])
        else:
            spin = np.array([-body.GetAngVelParent().y for body in self.b.particle_bodies])
        return self.b.get_velocities()+spin[:, None]*np.column_stack((-lever[:, 1], lever[:, 0]))

    def damp_preparation(self, dt):
        """Dissipative preparation only; never clips velocities during flow."""
        factor = math.exp(-self.s.STREAM_SETTLE_DAMPING*dt)
        if factor == 1.:
            return
        for body in self.b.particle_bodies:
            if self.pymunk:
                body.velocity *= factor
                body.angular_velocity *= factor
            else:
                body.SetPosDt(body.GetPosDt()*factor)
                body.SetAngVelLocal(body.GetAngVelLocal()*factor)

    def apply(self, i, buoyancy, force, point):
        body, s, b = self.b.particle_bodies[i], self.s, self.b
        if self.pymunk:
            body.apply_force_at_world_point((0., buoyancy*s.PX_PER_M), body.position)
            body.apply_force_at_world_point(tuple(force*s.PX_PER_M), tuple(point*s.PX_PER_M))
        else:
            c = b.chrono
            position = body.GetPos()
            accumulator = self.accumulators[i]
            if accumulator is None:
                body.EmptyAccumulators()
                body.AccumulateForce(c.ChVector3d(0, 0, buoyancy), position, False)
                body.AccumulateForce(c.ChVector3d(force[0], 0, force[1]), c.ChVector3d(point[0], position.y, point[1]), False)
            else:
                body.EmptyAccumulator(accumulator)
                body.AccumulateForce(accumulator, c.ChVector3d(0, 0, buoyancy), position, False)
                body.AccumulateForce(accumulator, c.ChVector3d(force[0], 0, force[1]),
                                     c.ChVector3d(point[0], position.y, point[1]), False)

    def open_outlet(self):
        b = self.b
        if self.pymunk:
            b.space.remove(b.walls[2], b.walls[3])
        else:
            b.sys.RemoveBody(b.wall_bodies[3])
            b.sys.RemoveBody(b.wall_bodies[1])

    def break_bonds(self, i):
        if not self.pymunk:
            return  # Chrono cohesion is a contact law, not a persistent spring.
        b, body = self.b, self.b.particle_bodies[i]
        for spring in list(b.bond_springs):
            if spring.a is body or spring.b is body:
                b.space.remove(spring)
                b.bond_springs.remove(spring)
                b._bond_break_distances.pop(spring, None)

    def grain_left(self, i, pos, angle):
        if self.pymunk:
            return self.s.px2m(self.b.particle_shapes[i].bb.left)
        local = self.b.particle_draw_local_vertices[i]
        if local is None:
            return pos[0]-self.b.particle_sizes[i]/2
        ca, sa = math.cos(angle), math.sin(angle)
        return pos[0]+float((np.asarray(local) @ np.array([[ca, sa], [-sa, ca]]))[:, 0].min())

    def recirculate(self, i, shift):
        b, body = self.b, self.b.particle_bodies[i]
        if self.pymunk:
            shape = b.particle_shapes[i]
            b.space.remove(shape)
            body.position = (body.position.x-self.s.m2px(shift), body.position.y)
            b.space.add(shape)
            b.space.reindex_shapes_for_body(body)
        else:
            p = body.GetPos()
            system = getattr(b, "sys", None)
            collision_system = system.GetCollisionSystem() if hasattr(system, "GetCollisionSystem") else None
            if collision_system is not None:
                body.RemoveCollisionModelsFromSystem(collision_system)
            body.SetPos(b.chrono.ChVector3d(p.x-shift, p.y, p.z))
            if collision_system is not None:
                body.SyncCollisionModels()
                body.AddCollisionModelsToSystem(collision_system)

    def suspend(self, i):
        """Wait outside the solver without discarding momentum or stable IDs."""
        b, body = self.b, self.b.particle_bodies[i]
        if self.pymunk:
            b.space.remove(b.particle_shapes[i], body)
            body.force, body.torque = (0, 0), 0
        else:
            b.sys.RemoveLink(b.stream_plane_links[i])
            b.sys.RemoveBody(body)

    def resume(self, i, shift):
        b, body = self.b, self.b.particle_bodies[i]
        if self.pymunk:
            body.position = (body.position.x-self.s.m2px(shift), body.position.y)
            b.space.add(body, b.particle_shapes[i])
            b.space.reindex_shapes_for_body(body)
        else:
            p = body.GetPos()
            body.SetPos(b.chrono.ChVector3d(p.x-shift, 0., p.z))
            b.sys.AddBody(body)
            # A previously initialized Chrono body retains its initialization
            # flag after RemoveBody. Re-add its Bullet proxy explicitly.
            body.SyncCollisionModels()
            body.AddCollisionModelsToSystem(b.sys.GetCollisionSystem())
            b.sys.AddLink(b.stream_plane_links[i])

    def remove(self, i):
        b, body = self.b, self.b.particle_bodies[i]
        if self.pymunk:
            b.space.remove(b.particle_shapes[i], body)
            body.velocity, body.angular_velocity = (0, 0), 0
            body.force, body.torque = (0, 0), 0
        else:
            b.sys.RemoveLink(b.stream_plane_links[i])
            b.sys.RemoveBody(body)
            body.SetPosDt(b.chrono.ChVector3d(0, 0, 0))
            body.SetAngVelLocal(b.chrono.ChVector3d(0, 0, 0))

    def kinetic_energy(self, active):
        energy = 0.
        for i in np.flatnonzero(active):
            body = self.b.particle_bodies[i]
            if self.pymunk:
                energy += .5*(body.mass*body.velocity.length_squared + body.moment*body.angular_velocity**2)
            else:
                v, w, inertia = body.GetPosDt(), body.GetAngVelLocal(), body.GetInertiaXX()
                energy += .5*(body.GetMass()*(v.x*v.x+v.y*v.y+v.z*v.z)
                              + inertia.x*w.x*w.x+inertia.y*w.y*w.y+inertia.z*w.z*w.z)
        return energy/self.s.PX_PER_M**2 if self.pymunk else energy


class OutletBarrier:
    """A fixed-width collider translated vertically to track the mean bed top.

    The prism extends below Z=0, so moving its top is sufficient and requires
    no mesh rebuilding. Height changes represent a quasistatic boundary update.
    """
    def __init__(self, s, backend, height):
        self.s, self.b = s, backend
        self.pymunk = hasattr(backend, "space")
        self.height = 0.
        self.depth = 2*s.DOMAIN_Z
        self.thickness = min(s.DOMAIN_X*.01, min(backend.particle_sizes)*.1)
        if self.pymunk:
            pm = backend.pymunk
            self.body = pm.Body(body_type=pm.Body.STATIC)
            self.shape = pm.Poly.create_box(self.body, (s.m2px(self.thickness), s.m2px(self.depth)))
            self.shape.friction = math.sqrt(s.FRICTION_COEFF)
            self.shape.elasticity = s.RESTITUTION
            self.shape.collision_type = 2
            backend.space.add(self.body, self.shape)
        else:
            self.body = backend.chrono.ChBodyEasyBox(self.thickness, backend._y_extent*2,
                                                    self.depth, 1000., False, True, backend._make_material(is_wall=True))
            self.body.SetFixed(True)
            backend.sys.AddBody(self.body)
        self.update(height)

    def update(self, height):
        self.height = float(np.clip(height, 0., self.s.DOMAIN_Z))
        x, z = self.s.DOMAIN_X+self.thickness/2, self.height-self.depth/2
        if self.pymunk:
            self.body.position = (self.s.m2px(x), self.s.m2px(z))
            self.b.space.reindex_shapes_for_body(self.body)
        else:
            self.body.SetPos(self.b.chrono.ChVector3d(x, 0, z))


class StreamForces:
    """Recompute upper-envelope exposure and apply traction with its moment arm.

    Exposure is the fraction of a grain's horizontal projection visible from
    above. Its equivalent exposed frontal height is fraction * grain height.
    Traction acts at the exposed cap centroid, buoyancy at the body centroid.
    Thus weight, friction, contacts and hydraulic moments are integrated by
    the rigid-body solver, including rolling over actual neighboring grains.
    """
    def __init__(self, s, backend):
        self.s, self.backend = s, backend
        self.physics = StreamPhysics(s, backend)
        self.active = np.ones(backend.num_particles(), dtype=bool)
        backend.stream_active = self.active
        self.pending = np.zeros_like(self.active)
        self.pending_shifts = np.zeros(len(self.active))
        self.queued_returns = 0
        self._footprint_radii = np.array([size/2 if local is None else float(np.linalg.norm(local, axis=1).max())
                                         for size, local in zip(backend.particle_sizes, backend.particle_draw_local_vertices)])
        self.exited = np.zeros_like(self.active)
        self.recirculations = np.zeros(len(self.active), dtype=np.int64)
        self.travel_offset = np.zeros(len(self.active))
        self.exposure = np.zeros(len(self.active))
        self.force = np.zeros((len(self.active), 2))
        self.points = backend.get_positions().copy()
        self.buoyancy = self.physics.masses * (
            s.WATER_DENSITY / s.PARTICLE_DENSITY * s.GRAVITY_MAG)
        self.full_buoyancy = self.buoyancy.copy()
        self.envelope_x = self.envelope_z = np.zeros(0)
        self.bed_surface_z = np.zeros(0)
        self.mean_bed_elevation = s.SLOPE_BASE
        self.barrier = None
        self.hydraulics = RiverHydraulics(s, backend)
        self.grain_velocity = np.zeros((len(self.active), 2))
        self.local_stress = np.zeros(len(self.active))
        self.current_stress = 0.
        self.drag_coefficient = np.zeros(len(self.active))
        self.drag_relative = np.zeros_like(self.force)
        self.lever = np.zeros_like(self.force)
        self.explicit_force = self.force.copy()

    def update(self, tau=None, *, time=None):
        s, b = self.s, self.backend
        pos, angles = b.get_positions(), b.get_angles()
        self.exposure.fill(0.)
        self.force.fill(0.)
        self.points = pos.copy()
        # At least 24 samples across the smallest grain. Midpoint samples avoid
        # circle tangencies and edge ambiguities; local slices bound cost.
        ncols = max(128, min(100000, int(math.ceil(s.DOMAIN_X / min(b.particle_sizes) * 24))))
        dx = s.DOMAIN_X / ncols
        xs = (np.arange(ncols) + .5) * dx
        profile = np.full(ncols, -np.inf)
        traces = []
        for i in np.flatnonzero(self.active):
            local = b.particle_draw_local_vertices[i]
            if local is None:
                r = b.particle_sizes[i]/2
                xmin, xmax = pos[i, 0]-r, pos[i, 0]+r
                zmin, zmax = pos[i, 1]-r, pos[i, 1]+r
            else:
                ca, sa = math.cos(angles[i]), math.sin(angles[i])
                verts = np.asarray(local) @ np.array([[ca, sa], [-sa, ca]]) + pos[i]
                xmin, zmin = verts.min(axis=0)
                xmax, zmax = verts.max(axis=0)
            start = max(0, int(math.ceil(xmin/dx-.5)))
            end = min(ncols, int(math.floor(xmax/dx-.5))+1)
            if end <= start:
                continue
            sample_x = xs[start:end]
            if local is None:
                top = pos[i, 1] + np.sqrt(np.maximum(0., r*r-(sample_x-pos[i, 0])**2))
            else:
                top = np.full(len(sample_x), -np.inf)
                for v, w in zip(verts, np.roll(verts, -1, axis=0)):
                    if abs(w[0]-v[0]) < 1e-14:
                        continue
                    u = (sample_x-v[0])/(w[0]-v[0])
                    hit = (u >= 0) & (u <= 1)
                    top[hit] = np.maximum(top[hit], v[1]+u[hit]*(w[1]-v[1]))
            profile[start:end] = np.maximum(profile[start:end], top)
            traces.append((i, start, end, top, zmin, zmax, xmax-xmin))
        for i, start, end, top, zmin, zmax, width in traces:
            visible = np.isfinite(top) & (top >= profile[start:end]-1e-9)
            # Use full width for partly exported grains, reducing in-domain forcing.
            fraction = min(1., visible.sum() * dx / max(width, dx))
            self.exposure[i] = fraction
            if not np.any(visible):
                continue
            exposed_height = (zmax-zmin)*fraction
            if tau is not None:
                self.force[i] = (tau*s.GRAIN_STRESS_FACTOR*exposed_height,
                                 tau*s.GRAIN_STRESS_FACTOR*exposed_height*s.STREAM_LIFT_RATIO)
                self.force[i] *= self.physics.area_width[i]
            self.points[i] = (xs[start:end][visible].mean(), zmax-exposed_height/2)
        self.envelope_x = xs
        self.envelope_z = np.where(np.isfinite(profile), profile, s.SLOPE_BASE)
        # Build the bed-connected surface from the floor upward. Detached
        # saltating grains contribute exposure, but do not raise the barrier.
        bed = np.full(ncols, s.SLOPE_BASE)
        gap_tolerance = max(min(b.particle_sizes)*.15, s.INITIAL_CLEARANCE*1.1)
        for i, start, end, top, zmin, zmax, width in sorted(traces, key=lambda trace: trace[4]):
            if zmin <= float(bed[start:end].max())+gap_tolerance:
                bed[start:end] = np.maximum(bed[start:end], top)
        self.bed_surface_z = bed
        mask = (xs >= s._wall_inset()) & (xs <= s.DOMAIN_X-s._wall_inset())
        self.mean_bed_elevation = float(bed[mask].mean()) if mask.any() else s.SLOPE_BASE
        if self.barrier is not None:
            self.barrier.update(self.mean_bed_elevation)
        if time is not None:
            h = self.hydraulics
            h.update(time, xs, bed, self.exposure, self.active)
            exposed_area = np.zeros(len(self.active))
            for i, start, end, top, zmin, zmax, width in traces:
                exposed_height = (zmax-zmin)*self.exposure[i]
                wet_height = float(np.clip(h.water_level-zmax+exposed_height, 0., exposed_height))
                exposed_area[i] = wet_height*self.physics.area_width[i]
                if wet_height > 0:
                    self.points[i, 1] = min(zmax, h.water_level)-wet_height/2
            # Exact circular segment / spherical cap submergence. Rotated
            # prisms use polygon clipping at the fixed water plane.
            self.buoyancy[:] = self.full_buoyancy
            for i in np.flatnonzero(self.active):
                local = b.particle_draw_local_vertices[i]
                if local is None:
                    r = b.particle_sizes[i]/2
                    height = float(np.clip(h.water_level-pos[i, 1]+r, 0, 2*r))
                    if self.physics.pymunk:
                        fraction = (r*r*math.acos(np.clip((r-height)/r, -1., 1.))
                                    -(r-height)*math.sqrt(max(0., 2*r*height-height*height)))/(math.pi*r*r)
                    else:
                        fraction = height*height*(3*r-height)/(4*r**3)
                else:
                    ca, sa = math.cos(angles[i]), math.sin(angles[i])
                    verts = np.asarray(local) @ np.array([[ca, sa], [-sa, ca]])+pos[i]
                    if verts[:, 1].max() <= h.water_level:
                        continue
                    clipped = []
                    for v, w in zip(verts, np.roll(verts, -1, axis=0)):
                        if v[1] <= h.water_level:
                            clipped.append(v)
                        if (v[1] <= h.water_level) != (w[1] <= h.water_level):
                            clipped.append(v+(w-v)*(h.water_level-v[1])/(w[1]-v[1]))
                    if len(clipped) < 3:
                        fraction = 0.
                    else:
                        polygon = np.asarray(clipped)
                        area = .5*abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                                     -np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
                        fraction = min(1., area/b.particle_planform_areas[i])
                self.buoyancy[i] *= fraction
            u, w, self.local_stress = h.sample(self.points)
            self.grain_velocity = np.column_stack((u, w))
            self.current_stress = float(np.trapezoid(h.tau, h.x)/s.DOMAIN_X)
            if s.HYDRAULIC_LOAD == "drag":
                self.lever = self.points-pos
                relative = self.grain_velocity-self.physics.point_velocity(self.lever)
                self.drag_relative = relative
                speed = np.linalg.norm(relative, axis=1)
                self.drag_coefficient = .5*s.WATER_DENSITY*s.STREAM_DRAG_COEFFICIENT*s.GRAIN_STRESS_FACTOR*exposed_area
                if s.SIMULATE_TURBULENCE and s.TURBULENCE_TARGET == "stress":
                    base = np.interp(self.points[:, 0], h.x, h.mean_tau)
                    self.drag_coefficient *= np.divide(self.local_stress, base, out=np.ones(len(base)), where=base > 0)
                self.drag_coefficient[self.points[:, 1] > h.water_level] = 0.
                self.force = self.drag_coefficient[:, None]*speed[:, None]*relative
                self.force[:, 1] += s.STREAM_LIFT_RATIO*np.abs(self.force[:, 0])
            else:
                horizontal = self.local_stress*s.GRAIN_STRESS_FACTOR*exposed_area
                self.force = np.column_stack((horizontal, s.STREAM_LIFT_RATIO*horizontal))
            # No fluid drag on grains above the specified water surface.
            self.force[self.points[:, 1] > h.water_level] = 0.
            if not np.isfinite(self.force).all() or not np.isfinite(h.tau).all():
                raise ValueError("Hydraulic loads are non-finite. Reduce velocity, stress or turbulence amplitude.")
            self.explicit_force = self.force.copy()

    def prepare_step(self, dt):
        """Effective, torque-aware drag force for the proposed substep."""
        s = self.s
        if s.HYDRAULIC_LOAD == "drag" and s.STREAM_DRAG_INTEGRATION == "implicit":
            self.force = implicit_drag_force(self.drag_relative, self.drag_coefficient,
                                             self.physics.masses, self.physics.inertias, self.lever, dt)
            self.force[:, 1] += s.STREAM_LIFT_RATIO*np.abs(self.force[:, 0])
        else:
            self.force = self.explicit_force.copy()
        return self.force

    def open_outlet(self):
        self.update(0.)
        self.physics.open_outlet()
        self.barrier = OutletBarrier(self.s, self.backend, self.mean_bed_elevation)

    def apply(self, hydraulic=True):
        for i in np.flatnonzero(self.active):
            self.physics.apply(i, self.buoyancy[i], self.force[i] if hydraulic else np.zeros(2), self.points[i])

    def remove_exported(self):
        """Export or recirculate outgoing grains, keeping stable particle IDs."""
        b, s = self.backend, self.s
        positions, angles = b.get_positions(), b.get_angles()
        # Try queued grains first, in stable ID order. The same target X/Z,
        # orientation and momentum are retained until their inlet space clears.
        for i in np.flatnonzero(self.pending):
            shift = self.pending_shifts[i]
            candidate = positions[i]-[shift, 0.]
            if self._inlet_clear(i, candidate, positions, angles):
                self.physics.resume(i, shift)
                self.travel_offset[i] += shift
                positions[i] = candidate
                self.pending[i], self.active[i] = False, True
        for i in np.flatnonzero(self.active):
            left = self.physics.grain_left(i, positions[i], angles[i])
            if left <= s.DOMAIN_X:
                continue
            self.physics.break_bonds(i)
            if s.RECIRCULATION:
                travel = s.DOMAIN_X-s._wall_inset()
                passes = max(1, int(math.ceil((left-s.DOMAIN_X)/travel)))
                shift = travel*passes
                candidate = positions[i]-[shift, 0.]
                if self._inlet_clear(i, candidate, positions, angles):
                    self.physics.recirculate(i, shift)
                    self.travel_offset[i] += shift
                    positions[i] = candidate
                else:
                    self.physics.suspend(i)
                    self.active[i], self.pending[i] = False, True
                    self.pending_shifts[i] = shift
                    self.queued_returns += 1
                self.recirculations[i] += passes
                continue
            self.physics.remove(i)
            self.active[i] = False
            self.exited[i] = True

    def _inlet_clear(self, i, candidate, positions, angles):
        """Exact circle checks, conservative polygon checks before teleporting."""
        b = self.backend
        radii = self._footprint_radii
        distance = np.linalg.norm(positions-candidate, axis=1)
        clearance = max(1e-9, min(b.particle_sizes)*.002)
        near = self.active & (distance < radii+radii[i]+clearance)
        near[i] = False
        if not np.any(near):
            return True
        if b.particle_draw_local_vertices[i] is None and all(b.particle_draw_local_vertices[j] is None for j in np.flatnonzero(near)):
            return False
        from shapely.geometry import Point, Polygon
        def footprint(index, center):
            local = b.particle_draw_local_vertices[index]
            if local is None:
                # Slightly circumscribed polygon avoids missed circle contact.
                return Point(center).buffer(radii[index]/math.cos(math.pi/64), quad_segs=16)
            ca, sa = math.cos(angles[index]), math.sin(angles[index])
            return Polygon(np.asarray(local) @ np.array([[ca, sa], [-sa, ca]])+center)
        target = footprint(i, candidate).buffer(clearance)
        return not any(target.intersects(footprint(j, positions[j])) for j in np.flatnonzero(near))

    def kinetic_energy(self):
        """Translation plus rotation, excluding bodies that have left the domain."""
        return self.physics.kinetic_energy(self.active)


def run_stream_simulation(s, progress_callback=None, save_video=False, return_data=True,
                          initialization_callback=None):
    validate_parameters(s)
    schedule = FrameSchedule(s, s.STREAM_DURATION)
    initialization_start = time.perf_counter()
    backend = s._initialize_backend(initialization_callback)
    n = backend.num_particles()
    if n == 0:
        raise ValueError("No sediment particles were generated. Check geometry and packing parameters.")
    forcing = StreamForces(s, backend)
    # Settle submerged sediment with the outlet closed, outside the flow clock.
    settle_steps = int(math.ceil(s.SETTLE_TIME / s.DT))
    settling, last_settle_report = 0., -math.inf
    while settling < s.SETTLE_TIME-1e-14:
        dt = computational_dt(s, backend, s.SETTLE_TIME-settling, enforce_safety=True,
                              travel_fraction=s.STREAM_CONTACT_TRAVEL_FRACTION)
        forcing.apply(hydraulic=False)
        backend.step(dt)
        forcing.physics.damp_preparation(dt)
        if hasattr(backend, "check_stream_boundaries"):
            backend.check_stream_boundaries(forcing.active)
        settling += dt
        if initialization_callback and settling-last_settle_report >= .1:
            last_settle_report = settling
            if initialization_callback(dict(cycle=1, max_cycles=1, phase="submerged settling",
                                            particle_count=n, completed_steps=min(settle_steps, int(round(settling/s.DT))),
                                            total_steps=settle_steps)) is False:
                raise s.HotstartCancelled("Stream-bed initialization cancelled.")
    init_pos = backend.get_positions().copy()
    backend.initial_positions = init_pos.tolist()
    if s.MATERIAL_TYPE == "cohesive" and hasattr(backend, "setup_bonds"):
        backend.setup_bonds()
    forcing.open_outlet()
    sizes = np.asarray(backend.particle_sizes)
    areas = np.asarray(backend.particle_planform_areas)
    moved = np.zeros(n, dtype=bool)
    frames = {k: [] for k in ("pos", "vel", "block_ang", "active", "exposure", "force", "time",
                              "stress", "pct", "exit", "ke", "coord", "bed_area",
                              "recirculated", "displacement", "barrier_height", "hydraulics", "dt",
                              "pending", "queued", "pos3d")}
    steps = int(math.ceil(s.STREAM_DURATION/s.DT))
    cancelled = False
    calculation_start = time.perf_counter()
    initialization_seconds = calculation_start-initialization_start
    last_progress = calculation_start
    actual_timesteps = []

    def record_frame(t, pos):
        frames["pos"].append(pos.copy())
        frames["vel"].append(backend.get_velocities().copy())
        frames["block_ang"].append(backend.get_angles().copy())
        frames["active"].append(forcing.active.copy())
        frames["pending"].append(forcing.pending.copy())
        frames["queued"].append(int(forcing.pending.sum()))
        frames["pos3d"].append(backend.get_positions3d() if hasattr(backend, "get_positions3d") else
                               np.column_stack((pos[:, 0], np.zeros(n), pos[:, 1])))
        frames["exposure"].append(forcing.exposure.copy())
        frames["force"].append(forcing.force.copy())
        frames["time"].append(t)
        frames["stress"].append(forcing.current_stress)
        frames["hydraulics"].append(forcing.hydraulics.snapshot())
        frames["dt"].append(actual_timesteps[-1] if actual_timesteps else 0.)
        frames["pct"].append(100*moved.mean())
        frames["exit"].append(int(forcing.exited.sum()))
        frames["recirculated"].append(int(forcing.recirculations.sum()))
        unwrapped = pos-init_pos
        unwrapped[:, 0] += forcing.travel_offset
        frames["displacement"].append(np.linalg.norm(unwrapped, axis=1))
        frames["ke"].append(forcing.kinetic_energy())
        coord, _ = backend.get_contact_info()
        frames["coord"].append(coord*n/max(1, forcing.active.sum()))
        frames["bed_area"].append(float(areas[forcing.active].sum()))
        frames["barrier_height"].append(forcing.barrier.height)

    t, computational_step = 0., 0
    while True:
        if hasattr(backend, "check_stream_boundaries"):
            backend.check_stream_boundaries(forcing.active)
        forcing.update(time=t)
        pos = backend.get_positions()
        if not np.isfinite(pos).all():
            raise ValueError("Stream solver produced non-finite positions. Reduce DT or hydraulic stress.")
        moved |= (pos[:, 0]+forcing.travel_offset-init_pos[:, 0] >= .5*sizes) | forcing.exited
        final = t >= s.STREAM_DURATION-1e-12
        if final:
            t = s.STREAM_DURATION
        dt = None
        if not final:
            # An equilibrium hydraulic refresh is not a transient-fluid CFL
            # restriction. Only physical time-varying forcing needs resolution.
            hydraulic_limit = forcing.hydraulics.correlation_time/10 if s.SIMULATE_TURBULENCE else math.inf
            if 0 <= t < s.STREAM_STARTUP_TIME:
                hydraulic_limit = min(hydraulic_limit, s.STREAM_STARTUP_TIME/20)
            if s.STREAM_FORCING == "harmonic":
                hydraulic_limit = min(hydraulic_limit, 1/(40*s.BED_STRESS_FREQUENCY))
            if s.HYDRAULIC_LOAD == "drag" and s.STREAM_DRAG_INTEGRATION == "explicit":
                relative = forcing.drag_relative
                coeff = np.linalg.norm(forcing.force, axis=1)/np.maximum(np.linalg.norm(relative, axis=1), 1e-12)
                response = np.divide(forcing.physics.masses, coeff, out=np.full(n, np.inf), where=coeff > 0)
                hydraulic_limit = min(hydraulic_limit, .2*float(response[forcing.active].min()) if forcing.active.any() else math.inf)
            # Safety substeps also protect fixed-DT runs from explicit-drag
            # instability and tunneling; output targets never shorten dt.
            candidate = min(s.DT, s.STREAM_DURATION-t, hydraulic_limit)
            # Evaluate the stable drag impulse at the proposed dt, not the
            # arbitrarily stiff instantaneous explicit force. Iteration only
            # reduces dt and retains genuine translation/geometry safeguards.
            for _ in range(32):
                forcing.prepare_step(candidate)
                allowed = computational_dt(s, backend, s.STREAM_DURATION-t, forcing.force,
                                           forcing.active, hydraulic_limit, enforce_safety=True,
                                           travel_fraction=s.STREAM_CONTACT_TRAVEL_FRACTION)
                if allowed >= candidate*(1-1e-7):
                    dt = candidate
                    break
                candidate = allowed
            else:
                raise ValueError("Stable drag/timestep selection did not converge. Reduce maximum DT or forcing.")
        saved = schedule.due(computational_step, t, None if final else min(s.STREAM_DURATION, t+dt))
        if saved:
            record_frame(t, pos)
        now = time.perf_counter()
        if progress_callback and (saved or now-last_progress >= .15):
            last_progress = now
            if progress_callback(min(steps, int(round(t/s.DT))), steps, t, 0., 100*moved.mean(), float("nan")) is False:
                if not saved and (schedule.contains(t) or not frames["time"]):
                    record_frame(t, pos)
                cancelled = True
                break
        if final:
            break
        forcing.apply()
        backend.step(dt)
        actual_timesteps.append(dt)
        t = min(s.STREAM_DURATION, t+dt)
        computational_step += 1
        forcing.remove_exported()
    calculation_seconds = time.perf_counter()-calculation_start
    nf = len(frames["time"])
    times = np.asarray(frames["time"])
    mobile = np.flatnonzero(np.asarray(frames["pct"]) > 0)
    size_quantiles = np.quantile(sizes, [.16, .5, .84])
    summary = (f"STREAM BED — {s.STREAM_FORCING} hydraulic forcing\n"
               f"Engine: {s.ENGINE}; "
               + ("XZ section per metre of channel width\n" if forcing.physics.pymunk else "3D spheres/prisms constrained to the XZ plane\n")
               + f"Domain / bed scale factor: {s.STREAM_DOMAIN_SCALE:g}\n"
               f"Final mean-bed outlet barrier elevation: {forcing.barrier.height:.6g} m\n"
               f"Grains: {n}; calculated flow time: {t:.3f} / {s.STREAM_DURATION:g} s; "
               f"cancelled: {cancelled}\n"
               f"Recirculation: {s.RECIRCULATION}; returned passages: {int(forcing.recirculations.sum())}; "
               f"waiting for clear inlet space: {int(forcing.pending.sum())}; total queued events: {forcing.queued_returns}\n"
               f"Particle size input (m): {s.PARTICLE_SIZE_INPUT or s.PARTICLE_SIZE}\n"
               f"Realized D16/D50/D84 (m, by count): {', '.join(f'{q:.6g}' for q in size_quantiles)}\n"
               f"Calculation wall time: {calculation_seconds:.2f} s; preparation: {initialization_seconds:.2f} s\n"
               f"Bed top/base/length: {s.SLOPE_HEIGHT:g}/{s.SLOPE_BASE:g}/{s.SLOPE_LENGTH:g} m\n"
               + (f"Stress input: {s.STREAM_FORCING}, mean {s.BED_STRESS_MEAN:g} Pa, "
                  f"amplitude {s.BED_STRESS_AMPLITUDE:g} Pa, frequency {s.BED_STRESS_FREQUENCY:g} Hz\n"
                  if s.STREAM_FORCING != "velocity" else
                  f"Mean-velocity input: {s.FLOW_MEAN_VELOCITY:g} m/s; unit discharge {s.FLOW_MEAN_VELOCITY*s.FLOW_DEPTH:g} m2/s\n")
               + f"Final spatial mean stress: {forcing.current_stress:.6g} Pa; "
               f"mean u*: {float(forcing.hydraulics.ustar.mean()):.6g} m/s\n"
               f"Grain stress multiplier: {s.GRAIN_STRESS_FACTOR:g}; lift/drag: {s.STREAM_LIFT_RATIO:g}\n"
               f"Water/sediment density: {s.WATER_DENSITY:g}/{s.PARTICLE_DENSITY:g} kg/m³\n"
               f"Cumulative mobile grains: {100*moved.mean():.1f}% (downstream travel >= 0.5 grain sizes)\n"
               f"Exported grains: {int(forcing.exited.sum())}; in-domain solid area: {float(areas[forcing.active].sum()):.6g} m²\n"
               + (("Hydraulic force/width = stress × multiplier × exposed frontal height.\n" if forcing.physics.pymunk else
                   "Hydraulic force = stress × multiplier × equivalent exposed frontal area.\n") if s.HYDRAULIC_LOAD == "traction" else
                  "Hydraulic drag = 0.5 rho Cd exposed_area |fluid_velocity-grain_velocity| (fluid_velocity-grain_velocity).\n")
               +
               "Buoyancy uses submerged volume (area per width in 2D). The contact solver balances\n"
               "weight, friction, hydraulic forces and their moments in each grain pocket.\n"
               "Exposure uses sampled top-envelope visibility and an equivalent exposed cap.\n"
               "Mobility is a reporting criterion; it does not trigger or constrain motion.\n"
               f"Hydraulics: {s.HYDRAULIC_SOLVER}; force closure: {s.HYDRAULIC_LOAD}; "
               f"inlet mean velocity {s.FLOW_MEAN_VELOCITY:g} m/s, reference depth {s.FLOW_DEPTH:g} m.\n"
               f"Surface D84={forcing.hydraulics.d84:.6g} m; ks/D84={s.ROUGHNESS_D84_FACTOR:g}; "
               f"z0={forcing.hydraulics.z0:.6g} m.\n"
               f"Hydraulic grid: {s.FLOW_COLUMNS} x {s.FLOW_LEVELS}; refresh {s.FLOW_UPDATE_INTERVAL:g} s; "
               f"final discharge relative error {forcing.hydraulics.discharge_error:.3g}, "
               f"momentum relative residual {forcing.hydraulics.momentum_error:.3g}.\n"
               f"Final mean Reynolds number: {float(forcing.hydraulics.reynolds_number.mean()):.6g}; "
               f"Froude: {float(forcing.hydraulics.froude_number.mean()):.6g}; "
               f"roughness Reynolds ks+: {float(forcing.hydraulics.roughness_reynolds_number.mean()):.6g}.\n"
               f"Turbulence: {s.SIMULATE_TURBULENCE}, field={s.TURBULENCE_TARGET}, seed={s.TURBULENCE_SEED}.\n"
               f"Startup ramp: {s.STREAM_STARTUP_TIME:g} s; settling damping: {s.STREAM_SETTLE_DAMPING:g} 1/s "
               f"(preparation only); drag integration: {s.STREAM_DRAG_INTEGRATION}.\n"
               f"Stream contact travel fraction: {s.STREAM_CONTACT_TRAVEL_FRACTION:g} of local grain thickness.\n"
               f"Adaptive dt: {s.ADAPTIVE_DT}; {len(actual_timesteps)} flow steps; "
               f"range {min(actual_timesteps, default=0):.6g} to {max(actual_timesteps, default=0):.6g} s.\n"
               "Hydraulics are equilibrium 1DV columns, with fixed stage and discharge in velocity mode.\n"
               "No streamwise momentum/free-surface solve, resolved eddies or HEC-RAS coupling.\n"
               "Synthetic fluctuations and drag/traction/lift closures need experimental calibration.\n"
               "Barrier height follows the bed-connected mean surface.\n"
               + ("Loads are N/m and KE is J/m.\n" if forcing.physics.pymunk else "Loads are N and KE is J; sphere/prism projected area is used.\n"))
    if getattr(backend, "engine_fallback_reason", None):
        summary += f"Requested {backend.requested_engine}; used {s.ENGINE}: {backend.engine_fallback_reason}\n"
    summary += "Numerical controls: " + ", ".join(f"{k}={getattr(s, k)}" for k in s._SOLVER_FIELDS) + "\n"
    summary += (f"Recording: {schedule.mode}; {nf} frames, actual saved window {times[0]:.6g} to {times[-1]:.6g} s; "
                f"requested {schedule.start:g} to {schedule.stop:g} s.\n"
                "Second targets use the nearest actual solver state (ties choose earlier); duplicate frames are merged.\n")
    summary += ("Stream grains are constrained to XZ and backed by lateral colliders; numerical floor/lateral escape stops the run.\n"
                "Recirculation waits outside the contact solver if the inlet is occupied, retaining grain momentum.\n"
                "Motion/drag safety substeps also protect fixed-DT runs; recording never forces a smaller timestep.\n")
    data = dict(stream_bed_setup=True, n_particles=n, init_pos=init_pos, sizes_static=sizes,
                c_sizes_static=np.asarray(backend.particle_c_sizes), planform_static=areas,
                draw_local_vertices=backend.particle_draw_local_vertices, is_dfn_mode=False,
                dfn_blocks=[], particle_shape=s.PARTICLE_SHAPE, particle_kinds=backend.particle_kinds,
                b_sizes_static=np.asarray(backend.particle_b_sizes), rect_b=s._rect_b, rect_c=s._rect_c,
                domain_x=s.DOMAIN_X, domain_y=s.DOMAIN_Y, domain_z=s.DOMAIN_Z,
                slope_height=s.SLOPE_HEIGHT, slope_base=s.SLOPE_BASE, slope_length=s.SLOPE_LENGTH,
                total_rotation=0., settle_time=s.SETTLE_TIME, sim_duration=s.STREAM_DURATION, fps=s.FPS,
                frames_pos=frames["pos"], frames_vel=frames["vel"], frames_block_ang=frames["block_ang"],
                frames_active=frames["active"], frames_exposure=frames["exposure"], frames_force=frames["force"],
                frames_pending=frames["pending"], arr_pending=np.asarray(frames["queued"]),
                frames_pos3d=frames["pos3d"],
                arr_total_retained_area=np.asarray([float(areas[a | p].sum()) for a, p in zip(frames["active"], frames["pending"])]),
                frames_displacement=frames["displacement"], recirculation=s.RECIRCULATION,
                frames_hydraulics=frames["hydraulics"], arr_dt=np.asarray(frames["dt"]),
                computational_timesteps=np.asarray(actual_timesteps),
                solver_parameters={k: getattr(s, k) for k in s._SOLVER_FIELDS},
                recording_parameters=schedule.metadata(), calculated_time=t,
                final_positions=backend.get_positions().copy(),
                final_mobile_percent=100*float(moved.mean()), final_exported_grains=int(forcing.exited.sum()),
                final_pending_grains=int(forcing.pending.sum()),
                arr_recirculated=np.asarray(frames["recirculated"]),
                particle_recirculations=forcing.recirculations.copy(),
                calculation_wall_seconds=calculation_seconds, initialization_wall_seconds=initialization_seconds,
                particle_size_input=s.PARTICLE_SIZE_INPUT or str(s.PARTICLE_SIZE),
                realized_size_quantiles=size_quantiles,
                engine=s.ENGINE, energy_unit="J/m" if forcing.physics.pymunk else "J",
                requested_engine=getattr(backend, "requested_engine", s.ENGINE),
                engine_fallback_reason=getattr(backend, "engine_fallback_reason", None),
                force_unit="N/m" if forcing.physics.pymunk else "N",
                domain_scale=s.STREAM_DOMAIN_SCALE, reference_cell_size=s.CELL_SIZE,
                arr_barrier_height=np.asarray(frames["barrier_height"]),
                arr_time=times, arr_angle=np.zeros(nf), arr_pct=np.asarray(frames["pct"]),
                arr_stress=np.asarray(frames["stress"]), arr_exit=np.asarray(frames["exit"]),
                arr_coord=np.asarray(frames["coord"]), arr_ke=np.asarray(frames["ke"]),
                arr_bed_area=np.asarray(frames["bed_area"]),
                failure_idx=int(mobile[0]) if len(mobile) else None, summary_text=summary,
                cancelled=cancelled, gravity_packed=s.GRAVITY_PACKED, fill_void=s.FILL_VOID,
                hotstart_pack=s.HOTSTART_PACK, hotstart_info=s._HOTSTART_INFO,
                initial_mixture=s._MIXED_PACKING_INFO,
                stream_parameters={k: getattr(s, k) for k in s._STREAM_FIELDS})
    data["arr_vel"] = np.asarray(frames["vel"])
    data["vel_mag_max"] = max(.001, float(np.max(np.linalg.norm(data["arr_vel"], axis=2))))
    for key in ("arr_fs", "arr_fa", "arr_mp", "arr_bd", "arr_touch", "arr_mob_small",
                "arr_mob_medium", "arr_mob_large"):
        data[key] = np.full(nf, np.nan)
    print(summary)
    if save_video:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        fig = plt.figure(figsize=(12, 7))
        anim = FuncAnimation(fig, lambda i: draw_stream_frame(fig, data, i), frames=nf, blit=False)
        try:
            anim.save(s.VIDEO_FILE, writer=FFMpegWriter(fps=s.FPS, bitrate=2500))
        finally:
            plt.close(fig)
    return data if return_data else None


def draw_water(ax, x, z, floor, ceiling, width, t, hydraulics=None):
    """Water occupies all pore space; opaque grain patches mask the solid area.

    Filling only above a top envelope leaves erroneous dry vertical shadows
    beneath overhangs/saltating grains. The backdrop must precede solids.
    """
    profile = np.clip(z, floor, ceiling)
    level_top = min(ceiling, hydraulics["water_level"]) if hydraulics else ceiling
    water = ax.fill_between([0., width], floor, level_top, color="#329fe8", alpha=.22, linewidth=0, zorder=0)
    water.set_gid("pore-water-background")
    if hydraulics:
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap
        hx, eta = hydraulics["x"], hydraulics["eta"]
        xx = np.broadcast_to(hx[:, None], hydraulics["velocity"].shape)
        zz = hydraulics["bed"][:, None]+hydraulics["depth"][:, None]*eta
        node_u = hydraulics["velocity"]
        cell_u = .25*(node_u[:-1, :-1]+node_u[1:, :-1]+node_u[:-1, 1:]+node_u[1:, 1:])
        palette = colormaps["Blues"](np.linspace(0., 1., 256))
        palette[:, :3] = .45*palette[:, :3]+.55*np.array([.82, .92, .98])
        # Explicit cell edges support a nonmonotonic, evolving bed without
        # inferred-edge warnings or alpha-blended Gouraud triangle seams.
        ax.pcolormesh(xx, zz, np.maximum(0., cell_u), shading="flat",
                      edgecolors="none", linewidth=0, antialiased=False,
                      cmap=ListedColormap(palette), vmin=0.,
                      vmax=max(.01, float(np.max(np.abs(hydraulics["velocity"])))), zorder=.2)
        selx = np.arange(0, len(hx), max(1, len(hx)//8))
        selz = np.arange(3, len(eta)-1, max(1, len(eta)//7))
        ix = np.ix_(selx, selz)
        ax.quiver(xx[ix], zz[ix], hydraulics["velocity"][ix], hydraulics["vertical_velocity"][ix],
                  color="#167bb5", alpha=.7, scale_units="xy",
                  scale=max(1., float(np.max(np.abs(hydraulics["velocity"]))))*10/width,
                  width=.002, zorder=.4)
        return
    level = max(float(np.percentile(profile, 90)) if len(profile) else floor, floor)
    row = min(ceiling-.08*(ceiling-floor), level+.55*(ceiling-level))
    for k in range(5):
        head = ((k/5 + .07*t) % 1)*width
        ax.annotate("", xy=(head, row), xytext=(max(0, head-.09*width), row),
                    arrowprops=dict(arrowstyle="->", color="#167bb5", alpha=.65, lw=1.3), zorder=.4)


def export_stream_csv(d, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "flow_time_s", "bed_stress_N_m2", "cumulative_mobile_pct",
                         "exported_grains", "remaining_grains",
                         "in_domain_KE_J_m" if d.get("energy_unit", "J/m") == "J/m" else "in_domain_KE_J",
                         "coordination_number", "remaining_solid_area_m2", "recirculated_passages",
                         "outlet_barrier_elevation_m", "friction_velocity_m_s", "roughness_z0_m",
                         "surface_D84_m", "unit_discharge_m2_s", "last_computational_dt_s",
                         "turbulence_active", "discharge_relative_error", "momentum_relative_error",
                         "queued_for_recirculation", "total_retained_solid_area_m2", "max_abs_lateral_Y_m",
                         "flow_startup_factor"])
        for i, t in enumerate(d["arr_time"]):
            h = d.get("frames_hydraulics", [None]*len(d["arr_time"]))[i]
            writer.writerow([i, t, d["arr_stress"][i], d["arr_pct"][i], d["arr_exit"][i],
                             int(d["frames_active"][i].sum()), d["arr_ke"][i],
                             d["arr_coord"][i], d["arr_bed_area"][i],
                             d.get("arr_recirculated", np.zeros(len(d["arr_time"]), dtype=int))[i],
                             d.get("arr_barrier_height", np.full(len(d["arr_time"]), np.nan))[i],
                             float(h["ustar"].mean()) if h else np.nan,
                             h["z0"] if h else np.nan, h["d84"] if h else np.nan,
                             float(h["discharge"].mean()) if h else np.nan,
                             d.get("arr_dt", np.zeros(len(d["arr_time"])))[i],
                             h["turbulence_active"] if h else False,
                             h["discharge_error"] if h else np.nan, h["momentum_error"] if h else np.nan,
                             d.get("arr_pending", np.zeros(len(d["arr_time"]), int))[i],
                             d.get("arr_total_retained_area", d["arr_bed_area"])[i],
                             float(np.max(np.abs(d["frames_pos3d"][i][d["frames_active"][i], 1])))
                             if "frames_pos3d" in d and d["frames_active"][i].any() else 0.,
                             h.get("startup_factor", 1.) if h else np.nan])


def draw_stream_frame(fig, d, idx, color_mode="auto", show_quiver=True):
    """Shared GUI/MP4 renderer; data snapshots make playback independent of controls."""
    from matplotlib.patches import Circle, Polygon
    from matplotlib import colormaps, colors
    fig.clear()
    ax = fig.add_axes([.07, .22, .56, .68])
    pos, active = d["frames_pos"][idx], d["frames_active"][idx]
    angles, sizes = d["frames_block_ang"][idx], d["sizes_static"]
    t = d["arr_time"][idx]
    speed = np.linalg.norm(d["frames_vel"][idx], axis=1)
    if color_mode == "size":
        values, maximum, label = sizes, max(sizes), "Grain size (m)"
    elif color_mode == "displacement":
        values = (d["frames_displacement"][idx] if "frames_displacement" in d else
                  np.linalg.norm(pos-d["init_pos"], axis=1))
        maximum, label = max(.001, max(values)), "Displacement (m)"
    else:
        values, maximum, label = speed, d["vel_mag_max"], "Grain speed (m/s)"
    cmap = colormaps["YlOrBr"]
    # Form the current free-surface profile using the same geometry as the forces.
    xs = np.linspace(0, d["domain_x"], 1000)
    surface = np.full(len(xs), d["slope_base"])
    for i in np.flatnonzero(active):
        lv = d["draw_local_vertices"][i]
        if lv is None:
            patch = Circle(pos[i], sizes[i]/2)
            mask = np.abs(xs-pos[i, 0]) <= sizes[i]/2
            surface[mask] = np.maximum(surface[mask], pos[i, 1]+np.sqrt(np.maximum(
                0, (sizes[i]/2)**2-(xs[mask]-pos[i, 0])**2)))
        else:
            ca, sa = math.cos(angles[i]), math.sin(angles[i])
            verts = np.asarray(lv) @ np.array([[ca, sa], [-sa, ca]]) + pos[i]
            patch = Polygon(verts)
            for v, w in zip(verts, np.roll(verts, -1, axis=0)):
                if abs(w[0]-v[0]) < 1e-14:
                    continue
                u = (xs-v[0])/(w[0]-v[0])
                mask = (u >= 0) & (u <= 1)
                surface[mask] = np.maximum(surface[mask], v[1]+u[mask]*(w[1]-v[1]))
        patch.set_facecolor(cmap(min(1., values[i]/max(maximum, 1e-9))*.8+.15))
        patch.set_edgecolor("#5a452b")
        patch.set_linewidth(.45)
        patch.set_zorder(2)
        ax.add_patch(patch)
    h = d.get("frames_hydraulics", [None]*len(d["arr_time"]))[idx]
    draw_water(ax, xs, surface, d["slope_base"], d["domain_z"], d["domain_x"], t, h)
    ax.plot([0, d["domain_x"]], [d["slope_base"]]*2, color=".3", lw=3)
    ax.plot([0, 0], [d["slope_base"], d["domain_z"]], color=".3", lw=3)
    ax.plot([0, d["slope_length"]], [d["slope_height"]]*2, "--", color=".4", lw=.8)
    if "arr_barrier_height" in d:
        height = d["arr_barrier_height"][idx]
        ax.plot([d["domain_x"]]*2, [0, height], color="#394b59", lw=4, zorder=6)
        ax.plot([d["domain_x"]-.015*d["domain_x"], d["domain_x"]+.015*d["domain_x"]],
                [height]*2, color="#394b59", lw=2, zorder=6)
    outlet_z = d.get("arr_barrier_height", np.full(len(d["arr_time"]), d["slope_base"]))[idx]
    ax.text(d["domain_x"], outlet_z+.04*d["domain_z"],
            "Recirculation → inlet" if d.get("recirculation") else "Outlet above barrier →", ha="right", fontsize=8)
    if show_quiver:
        sel = np.flatnonzero(active & (speed > .001))
        sel = sel[np.argsort(speed[sel])[-25:]]
        vel = d["frames_vel"][idx]
        ax.quiver(pos[sel, 0], pos[sel, 1], vel[sel, 0], vel[sel, 1],
                  color="#1d608a", scale_units="xy", scale=1., width=.003)
    ax.set(xlim=(-.03*d["domain_x"], 1.04*d["domain_x"]),
           ylim=(-.04*d["domain_z"], d["domain_z"]*1.02),
           xlabel="Streamwise X (m)", ylabel="Elevation Z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"Submerged bed • {d.get('engine', 'pymunk')} • domain scale {d.get('domain_scale', 1):g}", fontsize=10)
    norm = colors.Normalize(0, max(maximum, .001))
    # Match the restricted color range used above.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=ListedColormap(cmap(np.linspace(.15, .95, 256)))) ,
                        ax=ax, fraction=.035, pad=.025)
    cbar.set_label(label, fontsize=8)
    diagnostics = (("arr_stress", "Bed stress (N/m²)", "#167bb5"),
                   ("arr_pct", "Cumulative mobile (%)", "#a46722"),
                   (("arr_recirculated", "Recirculated passages", "#3e8556") if d.get("recirculation") else
                    ("arr_exit", "Exported grain count", "#3e8556")),
                   ("arr_ke", f"In-domain KE ({d.get('energy_unit', 'J/m')})", "#b34646"))
    for k, (key, title, color) in enumerate(diagnostics):
        a = fig.add_axes([.73, .73-k*.19, .24, .13])
        a.plot(d["arr_time"], d[key], color=color, alpha=.3)
        a.plot(d["arr_time"][:idx+1], d[key][:idx+1], color=color, lw=1.5)
        a.axvline(t, color=".4", lw=.7)
        a.set_title(title, fontsize=9)
        a.set_xlim(0, max(.01, d["arr_time"][-1]))
        a.tick_params(labelsize=7)
        a.grid(alpha=.2)
        if k == 3:
            a.set_xlabel("Flow time (s)", fontsize=8)
    transport = (f"recirculated = {d['arr_recirculated'][idx]}" if d.get("recirculation") else
                 f"exported = {d['arr_exit'][idx]}")
    if d.get("recirculation"):
        transport += f" | waiting at inlet = {d.get('arr_pending', np.zeros(len(d['arr_time']), int))[idx]}"
    fig.suptitle(f"Stream bed | t = {t:.2f} s | stress = {d['arr_stress'][idx]:.2f} Pa | "
                 f"mobile = {d['arr_pct'][idx]:.1f}% | {transport}", fontsize=11)
    fig.text(.07, .08, "Mobility: downstream travel ≥ 0.5 grain sizes. "
             + (f"Blue arrows: fluid velocity; u*={float(h['ustar'].mean()):.3g} m/s, z0={h['z0']:.3g} m.\n" if h else
                "Water arrows are visual only.\n")
                       + ("Grains return upstream with unchanged Z, velocity and spin." if d.get("recirculation") else
                          f"Forces: {d.get('force_unit', 'N/m')}. Grains can leave after clearing the mean-bed barrier."),
             fontsize=8, color=".35")
    return []
