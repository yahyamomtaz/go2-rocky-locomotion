"""Gait metrics from go2_sand.py logs: speed, base height, per-step foot sinkage and slip, mechanical power, CoT.

    uv run python scripts/go2_analyze.py runs/go2_rigid.npz runs/go2_sand_f075.npz

Stance: a foot carries more than 10 N of sand force (sand runs) or its bottom is within 5 mm of the ground (rigid).
Sinkage of a step: deepest point of the foot sphere below the undisturbed surface during that stance.
Slip of a step: horizontal travel of the foot centre from touchdown to liftoff (a planted foot does not move).
Power: sum over joints of |tau * qd|, tau from the PD law clipped to the effort limits. CoT = P / (m g v).
"""
import sys

import numpy as np

FEET = ("FL", "FR", "RL", "RR")


def stances(mask):
    """(start, end) index pairs of runs of True, dropping one-sample blips."""
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j < len(mask) and mask[j]:
                j += 1
            if j - k >= 2:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


def analyze(path, y_max=None):
    d = np.load(path)
    t, bq, bqd, feet = d["t"], d["base_q"], d["base_qd"], d["feet"]
    F, r, surf = d["foot_force"], float(d["foot_radius"]), float(d["surface"])
    rigid = bool(d["rigid"])
    walk = d["command"][:, 0] > 0
    w0 = np.argmax(walk) + int(1.0 / 0.02)   # skip the first second of walking
    w1 = len(t)
    if y_max is not None:
        beyond = np.nonzero(bq[:, 1] > y_max)[0]
        if len(beyond):
            w1 = min(w1, beyond[0])
    sl = slice(w0, w1)
    dur = t[w1 - 1] - t[w0]
    speed = (bq[w1 - 1, 1] - bq[w0, 1]) / dur
    height = np.mean(bq[sl, 2]) - surf
    bottom = feet[..., 2] - r - surf
    sink, slip, nsteps = [], [], 0
    for i in range(4):
        mask = (bottom[:, i] < 0.005) if rigid else (F[:, i, 2] > 10.0)
        for a, b in stances(mask):
            if a < w0 or b >= w1:
                continue
            nsteps += 1
            sink.append(-np.min(bottom[a:b, i]))
            slip.append(np.linalg.norm(feet[b - 1, i, :2] - feet[a, i, :2]))
    q, qd, tgt = d["joint_q"], d["joint_qd"], d["target"]
    tau = np.clip(d["ke"] * (tgt - q) - d["kd"] * qd, -d["effort_limit"], d["effort_limit"])
    power = np.mean(np.sum(np.abs(tau * qd), axis=1)[sl])
    mass = float(d["robot_mass"])
    cot = power / (mass * 9.81 * max(speed, 1e-3))
    return dict(name=path.split("/")[-1].replace(".npz", ""), rigid=rigid, friction=float(d["friction"]),
                cmd=float(d["speed"]), speed=speed, height=height, sink=np.mean(sink) if sink else 0.0,
                sink_max=np.max(sink) if sink else 0.0, slip=np.mean(slip) if slip else 0.0,
                steps_per_s=nsteps / dur, power=power, cot=cot, dur=dur,
                depth=float(d["sand_depth"]) if "sand_depth" in d else float(d["surface"]),
                cohesion=float(d["cohesion"]) if "cohesion" in d else 0.0,
                snow=str(d["snow"]) if "snow" in d else "",
                iters=int(d["mpm_iters"]) if "mpm_iters" in d else 500)


def summary(paths):
    """Mean and spread over repeated runs of each ground type (the GPU sand solve is not bitwise repeatable)."""
    groups = {}
    for p in paths:
        m = analyze(p, y_max=3.6)
        key = "rigid" if m["rigid"] else (f"{m['snow']} snow {m['depth'] * 100:.0f}cm" if m["snow"] else
                                          f"{np.degrees(np.arctan(m['friction'])):.0f}deg {m['depth'] * 100:.0f}cm"
                                          + (f" c{m['cohesion'] / 4000:.2g}kPa" if m["cohesion"] > 0 else "")
                                          + (f" it{m['iters']}" if m["iters"] != 500 else ""))
        groups.setdefault(key, []).append(m)
    keys = ("speed", "height", "sink", "sink_max", "slip", "power", "cot")
    print(f"{'ground':>18s} {'n':>2s} " + " ".join(f"{k:>16s}" for k in keys))
    for g, ms in groups.items():
        cells = []
        for k in keys:
            v = np.array([m[k] for m in ms]) * (1000 if k in ("height", "sink", "sink_max", "slip") else 1)
            cells.append(f"{v.mean():8.2f} +-{v.std(ddof=1) if len(v) > 1 else 0:6.2f}")
        print(f"{g:>18s} {len(ms):2d} " + " ".join(f"{c:>16s}" for c in cells))
        print(f"{'':>21s} speeds: " + ", ".join(f"{m['speed']:.2f}" for m in ms))
    print("speed m/s, height/sinkage/slip mm, power W; mean +- sample std over repeats; c = cohesion knob / 4000 Pa per kPa")


def main():
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
        return
    rows = [analyze(p, y_max=3.6) for p in sys.argv[1:]]
    print(f"{'run':22s} {'ground':>10s} {'cmd':>5s} {'speed':>6s} {'base h':>7s} {'sink':>6s} {'sink max':>8s} "
          f"{'slip':>6s} {'steps/s':>7s} {'power':>6s} {'CoT':>5s}")
    for m in rows:
        ground = ("rigid" if m["rigid"] else f"{m['snow']} snow" if m["snow"]
                  else f"sand {np.degrees(np.arctan(m['friction'])):.0f} deg")
        print(f"{m['name']:22s} {ground:>10s} {m['cmd']:5.2f} {m['speed']:6.2f} {m['height'] * 1000:6.0f}  "
              f"{m['sink'] * 1000:5.0f}  {m['sink_max'] * 1000:7.0f}  {m['slip'] * 1000:5.0f}  {m['steps_per_s']:6.1f}  "
              f"{m['power']:5.0f}  {m['cot']:5.2f}")
    print("speed m/s, base height above the sand surface mm, sinkage/slip mm per step, power W")


if __name__ == "__main__":
    main()
