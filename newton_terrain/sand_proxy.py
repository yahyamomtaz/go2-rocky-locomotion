"""Fast granular-contact model for the Go2 feet and shins, calibrated against the MPM sand (mpm_intruder_test.py).

Per leg, applied as an external wrench on the calf body (MuJoCo Warp integrates it with the robot):
- Foot bearing, elasto-plastic with footprint memory: the footprint floor sits at depth m below the surface. Pushing
  below it compresses a stiff spring; once that force exceeds the bearing capacity F_b(d) = A d^n at the current depth
  d, the floor yields and m deepens. Lifting the foot unloads it; a footprint is forgotten once the foot is 1 cm clear.
- Foot drag: sideways resistance A_h d^n_h opposing the horizontal foot velocity (rate independent, regularised at low
  speed so the explicit force stays stable), with an upward lift of lift_r times its magnitude (sand wedge).
- Shin and thigh drag: the buried length L of each leg rod resists its own sideways motion with S L^2 (passive earth
  pressure), so a leg sweeping through deep sand is held back.
- Hole memory: each world keeps its last 16 footprints; a foot landing in one starts at (keep x) its floor depth, so a
  robot that is not making progress keeps stepping into its own holes and digs itself in, as it does in the MPM sand.
- Pull-out: a small resistance P d while the foot moves up through the sand (cohesion makes it larger).
- Belly: if the base box reaches the sand it is held by a stiff spring with sideways friction.
The rigid floor (MuJoCo contact) is z = 0 in every world; the sand surface lies at z = surf[world] above it.
"""
import numpy as np
import warp as wp

R_FOOT = 0.022
R_SHIN = 0.013
K_FLOOR = 1.0e4   # footprint floor stiffness [N/m]
C_FLOOR = 60.0    # its damping while compressing [N s/m]
C_MAX = 400.0     # cap on the low-speed drag slope [N s/m] (explicit-force stability)


@wp.struct
class SandTable:
    bear_a: wp.array[float]
    bear_n: wp.array[float]
    drag_a: wp.array[float]
    drag_n: wp.array[float]
    lift_r: wp.array[float]
    shin_s: wp.array[float]
    pull_p: wp.array[float]


@wp.func
def rod_drag(p_lo: wp.vec3, p_hi: wp.vec3, zs: float, qd: wp.spatial_vector, com: wp.vec3, s: float, lift: float):
    """Passive-pressure drag s L^2 on the buried part (length L) of the rod p_lo -> p_hi, opposing the horizontal
    velocity of the buried midpoint (rigid-body velocity of the rod's body). Returns force and point of action."""
    F = wp.vec3(0.0)
    p = p_lo
    if p_lo[2] < zs and p_hi[2] > p_lo[2]:
        frac = wp.clamp((zs - p_lo[2]) / (p_hi[2] - p_lo[2]), 0.0, 1.0)
        L = frac * wp.length(p_hi - p_lo)
        p = p_lo + (p_hi - p_lo) * (0.5 * frac)
        v = wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), p - com)
        vh = wp.vec3(v[0], v[1], 0.0)
        fm = s * L * L
        veps = wp.max(0.1, fm / C_MAX)
        F = -fm * vh / wp.sqrt(wp.dot(vh, vh) + veps * veps)
        F = F + wp.vec3(0.0, 0.0, lift * wp.length(F))
    return F, p


@wp.kernel
def sand_forces(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                nb: int, calf_local: wp.array[int], foot_local: wp.array[int], thigh_local: wp.array[int],
                base_local: int,
                surf: wp.array[float], sand_type: wp.array[int], scale_b: wp.array[float], scale_h: wp.array[float],
                tab: SandTable, keep: wp.array[float], nh: int,
                hole_x: wp.array[float], hole_y: wp.array[float], hole_d: wp.array[float], hole_n: wp.array[int],
                mem: wp.array[float], contact: wp.array[float], depth_out: wp.array[float],
                body_f: wp.array[wp.spatial_vector]):
    tid = wp.tid()            # one thread per leg
    w = tid // 4
    leg = tid - 4 * w
    typ = sand_type[w]
    calf = w * nb + calf_local[leg]
    foot = w * nb + foot_local[leg]
    pf = wp.transform_get_translation(body_q[foot])
    vf = wp.spatial_top(body_qd[foot])
    zs = surf[w]                # sand surface height in this world (the rigid floor is at z = 0)
    d = zs - (pf[2] - R_FOOT)   # depth of the sphere bottom below the surface
    depth_out[tid] = d
    if typ < 0:                 # rigid ground world: MuJoCo handles everything
        mem[tid] = 0.0
        contact[tid] = wp.where(d > -0.004, 1.0, 0.0)
        return
    sb = scale_b[w]
    sh = scale_h[w]
    F = wp.vec3(0.0)
    m = float(mem[tid])
    fz = float(0.0)
    if d > 0.0 and m < 0.0:
        # touchdown (m < 0 marks "out of the sand"): start from the floor of an old hole if the foot lands in one
        m = float(0.0)
        for k in range(nh):
            hx = hole_x[w * nh + k]
            hy = hole_y[w * nh + k]
            dist = wp.sqrt((pf[0] - hx) * (pf[0] - hx) + (pf[1] - hy) * (pf[1] - hy))
            if dist < 2.0 * R_FOOT:
                m = wp.max(m, keep[w] * hole_d[w * nh + k] * (1.0 - dist / (2.0 * R_FOOT)))
    if d > 0.0:
        m = wp.max(m, 0.0)
        e = d - m
        if e > 0.0:
            fy = sb * tab.bear_a[typ] * wp.pow(d, tab.bear_n[typ])
            fe = K_FLOOR * e
            if fe > fy:
                m = d - fy / K_FLOOR
                fe = fy
            fz = fe + C_FLOOR * wp.max(-vf[2], 0.0)
        if vf[2] > 0.0:
            fz -= tab.pull_p[typ] * d * wp.min(vf[2] / 0.05, 1.0)
        vh = wp.vec3(vf[0], vf[1], 0.0)
        fh = sh * tab.drag_a[typ] * wp.pow(d, tab.drag_n[typ])
        veps = wp.max(0.1, fh / C_MAX)
        Fh = -fh * vh / wp.sqrt(wp.dot(vh, vh) + veps * veps)
        F = wp.vec3(Fh[0], Fh[1], fz + tab.lift_r[typ] * wp.length(Fh))
    elif d < -0.01 and m >= 0.0:
        # liftoff: remember this footprint (ring buffer shared by the 4 feet of the world)
        if m > 0.005:
            slot = wp.atomic_add(hole_n, w, 1) % nh
            hole_x[w * nh + slot] = pf[0]
            hole_y[w * nh + slot] = pf[1]
            hole_d[w * nh + slot] = m
        m = -1.0
    mem[tid] = m
    contact[tid] = wp.where(fz > 1.0, 1.0, 0.0)
    # shin: knee-to-foot rod, buried below the surface, dragged with its own velocity
    pk = wp.transform_get_translation(body_q[calf])
    X = body_q[calf]
    com = wp.transform_point(X, body_com[calf])
    Fs, ps = rod_drag(pf, pk, zs, body_qd[calf], com, sh * tab.shin_s[typ], tab.lift_r[typ])
    tau = wp.cross(pf - com, F) + wp.cross(ps - com, Fs)
    body_f[calf] = body_f[calf] + wp.spatial_vector(F + Fs, tau)
    # thigh: hip-to-knee rod, only once the knee is in the sand
    thigh = w * nb + thigh_local[leg]
    ph = wp.transform_get_translation(body_q[thigh])
    if pk[2] < zs:
        comt = wp.transform_point(body_q[thigh], body_com[thigh])
        Ft, pt = rod_drag(pk, ph, zs, body_qd[thigh], comt, sh * tab.shin_s[typ], tab.lift_r[typ])
        body_f[thigh] = body_f[thigh] + wp.spatial_vector(Ft, wp.cross(pt - comt, Ft))
    # belly (leg 0 thread does the base)
    if leg == 0:
        base = w * nb + base_local
        pb = wp.transform_get_translation(body_q[base])
        vb = wp.spatial_top(body_qd[base])
        db = zs - (pb[2] - 0.06)
        if db > 0.0:
            fzb = wp.max(3.0e4 * db + 300.0 * wp.max(-vb[2], 0.0), 0.0)
            vhb = wp.vec3(vb[0], vb[1], 0.0)
            Fb = -0.6 * fzb * vhb / wp.sqrt(wp.dot(vhb, vhb) + 0.01)
            body_f[base] = body_f[base] + wp.spatial_vector(wp.vec3(Fb[0], Fb[1], fzb), wp.vec3(0.0))


SAND_TYPES = ["f0.6", "f0.75", "f0.9", "f1.0", "f0.75_c2000", "f0.75_c4000", "f0.75_c8000"]


def fit_table(paths, r=R_FOOT):
    """Fit the proxy coefficients for each sand type from the MPM intruder tests."""
    rows = []
    for p in paths:
        d = np.load(p)
        s = float(d["surface"])
        dep = s - (d["plunge_p"][:, 2] - r)
        fz = d["plunge_f"][:, 2]
        m = (dep > 0.01) & (dep < (0.28 if "drags" in d and int(d["drags"][-1]) > 10 else 0.12)) & (fz > 1.0)
        bn, ba = np.polyfit(np.log(dep[m]), np.log(fz[m]), 1)
        dd, fh, lr = [], [], []
        for k in (d["drags"] if "drags" in d else (3, 6, 10)):
            f = d[f"drag{k}_f"]
            h = f[len(f) // 3:]
            dd.append(k / 100.0)
            fh.append(-h[:, 0].mean())
            lr.append(h[:, 2].mean() / max(-h[:, 0].mean(), 1e-6))
        hn, ha = np.polyfit(np.log(dd), np.log(fh), 1)
        shin = np.mean([-d[f"shin{L}_f"][len(d[f"shin{L}_f"]) // 3:, 0].mean() / (L / 100.0) ** 2
                        for L in (d["shins"] if "shins" in d else (5, 10))])
        pf = d["pull_f"][:, 2]
        pdep = s - (d["pull_p"][:, 2] - r)
        pm = pdep > 0.01
        pull = max(0.0, float(np.median(-pf[pm] / pdep[pm]))) if pm.any() else 0.0
        rows.append(dict(bear_a=float(np.exp(ba)), bear_n=float(bn), drag_a=float(np.exp(ha)), drag_n=float(hn),
                         lift_r=float(np.mean(lr)), shin_s=float(shin), pull_p=pull))
    return rows


def make_table(rows, device=None):
    t = SandTable()
    for k in ("bear_a", "bear_n", "drag_a", "drag_n", "lift_r", "shin_s", "pull_p"):
        setattr(t, k, wp.array(np.array([r[k] for r in rows], dtype=np.float32), dtype=float, device=device))
    return t
