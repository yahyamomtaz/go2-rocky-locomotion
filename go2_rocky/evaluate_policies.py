"""Paired, held-out evaluation through the existing simulation CLI.

Each policy receives the same terrain seed, geometry, initial pose, command,
and goal. The subprocess boundary releases all GPU allocations between runs.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import newton.utils

from go2_rocky.controller import PathControllerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True,
                        help="LABEL=PATH.onnx; LABEL=pretrained selects Newton's original actor")
    parser.add_argument("--terrain", choices=("smooth", "rocky", "mountain"), default="mountain")
    parser.add_argument("--seeds", type=int, nargs="+", default=(101, 102, 103))
    parser.add_argument("--slope", type=float, default=15)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--goal-distance", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="reuse complete matching logs; check settings and policy hash first")
    args = parser.parse_args()
    policies = []
    for specification in args.policy:
        label, separator, path = specification.partition("=")
        if not separator or not label or not all(c.isalnum() or c in "_-" for c in label):
            parser.error("--policy must be LABEL=PATH with a simple filename label")
        if path != "pretrained" and not Path(path).is_file():
            parser.error(f"policy missing: {path}")
        policies.append((label, path))
    if len(set(label for label, _ in policies)) != len(policies):
        parser.error("policy labels must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in args.seeds:
        initial = grid = geometry = None
        for label, path in policies:
            log = args.output / f"{label}_seed{seed}.npz"
            reuse = log.exists() and args.reuse_existing
            if log.exists() and not reuse:
                raise FileExistsError(f"evaluation already exists: {log}")
            command = [sys.executable, "-m", "go2_rocky.run_go2", "--terrain", args.terrain,
                       "--seed", str(seed), "--slope-deg", str(args.slope), "--roughness", "medium",
                       "--speed", str(args.speed), "--seconds", str(args.seconds), "--goal-distance",
                       str(args.goal_distance), "--viewer", "null", "--log", str(log)]
            if path != "pretrained":
                command.extend(("--policy", path))
            if not reuse:
                with log.with_suffix(".console.txt").open("w") as console:
                    subprocess.run(command, stdout=console, stderr=subprocess.STDOUT, check=True)
            with np.load(log) as data:
                meta = json.loads(data["metadata_json"].item())
                if reuse:
                    settings = {"terrain": args.terrain, "slope_deg": args.slope,
                                "speed_m_s": args.speed, "goal_distance_m": args.goal_distance,
                                "walking_seconds_requested": args.seconds, "direction": "uphill",
                                "friction": 0.75, "controller": "path"}
                    if any(meta.get(key) != value for key, value in settings.items()):
                        raise ValueError(f"existing evaluation settings differ: {log}")
                    terrain_config = meta.get("mountain_config") or meta.get("rock_config") or {}
                    if terrain_config.get("seed", seed) != seed:
                        raise ValueError(f"existing evaluation seed differs: {log}")
                    if terrain_config.get("roughness", "medium") != "medium":
                        raise ValueError(f"existing evaluation roughness differs: {log}")
                    if meta.get("controller_parameters") != asdict(PathControllerConfig()):
                        raise ValueError(f"existing evaluation controller differs: {log}")
                    if args.terrain == "mountain":
                        expected = {"size_m": 80, "height_m": 26, "resolution_m": 0.2,
                                    "surface_rockiness": 1, "trail_length_m": 40, "trail_half_width_m": 1}
                        if any(terrain_config.get(key) != value for key, value in expected.items()):
                            raise ValueError(f"existing mountain settings differ: {log}")
                    policy_file = (Path(newton.utils.download_asset("unitree_go2"))/"rl_policies/mjw_go2.onnx"
                                   if path == "pretrained" else Path(path))
                    if hashlib.sha256(policy_file.read_bytes()).hexdigest() != meta["policy_sha256"]:
                        raise ValueError(f"existing evaluation policy differs: {log}")
                if initial is None:
                    initial = data["joint_q"][0].copy()
                    grid = data["mountain_elevation_m"].copy() if "mountain_elevation_m" in data else None
                    geometry = meta["rocks"]
                else:
                    np.testing.assert_array_equal(data["joint_q"][0], initial)
                    if grid is not None:
                        np.testing.assert_array_equal(data["mountain_elevation_m"], grid)
                    if geometry != meta["rocks"]:
                        raise AssertionError("paired rock geometry differs")
            result = {"policy": label, "seed": seed, "log": str(log),
                      **{key: meta[key] for key in ("status", "path_progress_m", "completion_time_s",
                                                   "base_height_gain_m", "rock_contact_steps", "policy_sha256")}}
            cases.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "cases.json").write_text(json.dumps(cases, indent=2)+"\n")
    summary = [{"policy": label, "episodes": len(args.seeds),
                "successes": sum(c["status"] == "goal_reached" for c in cases if c["policy"] == label),
                "mean_progress_m": float(np.mean([c["path_progress_m"] for c in cases if c["policy"] == label])),
                "progress_std_m": float(np.std([c["path_progress_m"] for c in cases if c["policy"] == label])),
                "note": "Small paired diagnostic; not a broad reliability benchmark."}
               for label, _ in policies]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
