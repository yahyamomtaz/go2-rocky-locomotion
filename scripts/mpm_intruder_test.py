"""Force on a Go2 foot (2.2 cm sphere) or shin (1.3 cm vertical rod) moved along a prescribed path through MPM sand.

The intruder is kinematic: the sand solver sees it as a moving collider and the force is read back from the collider
impulses. These curves calibrate the fast sand-contact model used for training (newton_terrain/sand_proxy.py).

    uv run python scripts/mpm_intruder_test.py --friction 0.75 --out runs/intruder_f0.75.npz
Tests: plunge (down at 0.1 m/s to 12 cm), drag (sideways at 0.3 m/s at 3, 6, 10 cm), shin drag (rod buried 5 and
10 cm, sideways at 0.3 m/s), pull-out (from 8 cm, up at 0.2 m/s). Same MPM settings as go2_sand.py.
"""
import argparse

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM


@wp.kernel
def _body_force(impulse: wp.array[wp.vec3], cid: wp.array[int], collider_body: wp.array[int], body: int,
                inv_dt: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    c = cid[i]
    if c >= 0 and collider_body[c] == body:
        wp.atomic_add(out, 0, impulse[i] * inv_dt)


def run(a, shape, path, depth_bed=0.25, half=(0.35, 0.2)):
    depth_bed = a.bed_depth
    """path(t) -> (position, velocity) of the intruder centre; returns times, positions, forces on the intruder."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(b)
    p0, _ = path(0.0)
    body = b.add_body(xform=wp.transform(wp.vec3(*p0), wp.quat_identity()))
    # zero mass: the sand solver then treats the intruder as kinematic (it follows the path whatever the sand does)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=True)
    if shape == "foot":
        b.add_shape_sphere(body, radius=0.022, cfg=cfg)
    else:   # shin: vertical rod, 1.3 cm radius, 12 cm long, centred on the body
        b.add_shape_capsule(body, radius=0.013, half_height=a.shin_half, cfg=cfg)
    b.add_ground_plane()
    b.shape_flags[-1] = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    hx, hy = half
    lo, hi = np.array([-hx, -hy, 0.0]), np.array([hx, hy, depth_bed])
    wall = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=True)
    for x, y, sx, sy in ((-hx - 0.02, 0, 0.02, hy + 0.04), (hx + 0.02, 0, 0.02, hy + 0.04),
                         (0, -hy - 0.02, hx, 0.02), (0, hy + 0.02, hx, 0.02)):
        b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, depth_bed), wp.quat_identity()), hx=sx, hy=sy,
                        hz=depth_bed, cfg=wall)
    res = np.array(np.ceil(a.ppc * (hi - lo) / a.voxel), dtype=int)
    cell = (hi - lo) / res
    n0 = len(b.particle_q)
    b.add_particle_grid(pos=wp.vec3(lo), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=res[0] + 1,
                        dim_y=res[1] + 1, dim_z=res[2] + 1, cell_x=cell[0], cell_y=cell[1], cell_z=cell[2],
                        mass=float(np.prod(cell) * a.density), jitter=0.0, radius_mean=float(np.max(cell) * 0.5))
    rng = np.random.default_rng(a.seed)
    q = np.asarray(b.particle_q[n0:], dtype=np.float64)
    q += (rng.random(q.shape) - 0.5) * float(np.max(cell))
    b.particle_q[n0:] = q.tolist()
    model = b.finalize()
    model.mpm.friction.fill_(a.friction)
    if a.cohesion > 0:
        model.mpm.yield_stress.fill_(a.cohesion)
    if a.snow:
        model.mpm.yield_pressure.fill_(a.yield_pressure)
        model.mpm.hardening.fill_(a.hardening)
        model.mpm.tensile_yield_ratio.fill_(0.2)
        model.mpm.dilatancy.fill_(1.0)
    mpm = SolverImplicitMPM.Config()
    mpm.voxel_size, mpm.tolerance, mpm.grid_type = a.voxel, 1e-6, "sparse"
    mpm.strain_basis, mpm.max_iterations, mpm.critical_fraction = "P0", 500, 0.0
    mpm.velocity_basis, mpm.collider_basis = "Q1", "S2"
    solver = SolverImplicitMPM(model, mpm)
    if a.snow:
        model.mpm.particle_Jp.fill_(a.jp0)
    s0, s1 = model.state(), model.state()
    force = wp.zeros(1, dtype=wp.vec3)
    dt, T = 0.01, path.duration
    ts, ps, fs = [], [], []
    # let the bed settle under gravity with the intruder held in place
    for k in range(int(T / dt) + 30):
        t = max(0.0, (k - 30) * dt)
        p, v = path(t) if k >= 30 else path(0.0)
        if k < 30:
            v = np.zeros(3)
        bq = s0.body_q.numpy()
        bq[body] = [*p, 0.0, 0.0, 0.0, 1.0]
        s0.body_q.assign(bq)
        bqd = s0.body_qd.numpy()
        bqd[body] = [*v, 0.0, 0.0, 0.0]
        s0.body_qd.assign(bqd)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        force.zero_()
        imp, _pos, cid = solver.collect_collider_impulses(s0)
        wp.launch(_body_force, dim=imp.shape[0], inputs=[imp, cid, solver.collider_body_index, body, 1.0 / dt, force])
        if k >= 30:
            ts.append(t)
            ps.append(np.array(p))
            fs.append(force.numpy()[0].copy())
    return np.array(ts), np.array(ps), np.array(fs)


class Path:
    def __init__(self, fn, duration):
        self.fn, self.duration = fn, duration

    def __call__(self, t):
        return self.fn(min(t, self.duration))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", type=float, default=0.75)
    ap.add_argument("--cohesion", type=float, default=0.0)
    ap.add_argument("--density", type=float, default=1600.0)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--ppc", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snow", default="", choices=["", "settled", "powder", "packed"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.bed_depth = 0.25
    a.shin_half = 0.06
    drags, shins, plunge_to = (3, 6, 10), (5, 10), 0.12
    if a.snow:   # snow presets from go2_sand.py; feet sink much deeper, so test deeper
        from go2_sand import SNOW

        a.density, a.friction, a.yield_pressure, a.hardening, a.jp0, a.cohesion = SNOW[a.snow]
        a.bed_depth = 0.45
        a.shin_half = 0.12
        drags, shins, plunge_to = (3, 10, 20), (10, 20), 0.28
    wp.config.quiet = True
    D, r = a.bed_depth, 0.022
    surf = D + 0.0075   # settled surface sits about half a particle spacing above the grid top
    out = {}
    # plunge: sphere bottom from 1 cm above the surface to 12 cm deep at 0.1 m/s
    z0 = surf + r + 0.01
    out["plunge"] = run(a, "foot", Path(lambda t: (np.array([0.0, 0.0, z0 - 0.1 * t]), np.array([0, 0, -0.1])),
                                        (plunge_to + 0.01) / 0.1))
    # pull-out: start 8 cm deep (placed, then held while the bed settles), up at 0.2 m/s
    zc = surf - 0.08 + r
    out["pull"] = run(a, "foot", Path(lambda t: (np.array([0.0, 0.0, zc + 0.2 * t]), np.array([0, 0, 0.2])), 0.5))
    # drags: sphere centre at depth d below the surface, sideways at 0.3 m/s over 0.4 m
    for d in (x / 100.0 for x in drags):
        zc = surf - d + r
        out[f"drag{int(d * 100)}"] = run(a, "foot", Path(lambda t, zc=zc: (np.array([-0.2 + 0.3 * t, 0.0, zc]),
                                                                          np.array([0.3, 0, 0])), 1.3))
    # shin: rod bottom buried L below the surface, sideways at 0.3 m/s
    for L in (x / 100.0 for x in shins):
        zc = surf - L + 0.013 + a.shin_half
        out[f"shin{int(L * 100)}"] = run(a, "shin", Path(lambda t, zc=zc: (np.array([-0.2 + 0.3 * t, 0.0, zc]),
                                                                          np.array([0.3, 0, 0])), 1.3))
    np.savez(a.out, surface=surf, friction=a.friction, cohesion=a.cohesion, drags=np.array(drags), shins=np.array(shins),
             **{f"{k}_{n}": v for k, (t, p, f) in out.items() for n, v in (("t", t), ("p", p), ("f", f))})
    for k, (t, p, f) in out.items():
        s = len(t) // 2
        print(f"{k:8s} mid-test force (x, y, z) N: {np.round(f[s:].mean(0), 1)}  depth at mid {surf - (p[s, 2] - r):.3f} m",
              flush=True)


if __name__ == "__main__":
    main()
