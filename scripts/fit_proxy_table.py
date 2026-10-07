"""Fit the fast terrain model's coefficients from the MPM intruder tests and write them as a JSON table.

Reads runs/intruder_<type>.npz (from mpm_intruder_test.py) for each sand and snow type found and writes one row of
coefficients per type (bearing, drag, lift, shin drag, pull-out). The drag lift is zeroed for snow: the policy
learned to ride on it.

    uv run python scripts/fit_proxy_table.py
"""
import argparse
import json
import os

from newton_terrain import sand_proxy as SP

SNOW_TYPES = ["snow_settled", "snow_powder", "snow_packed"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="folder with the intruder_<type>.npz files")
    p.add_argument("--out", default="runs/proxy_table.json")
    args = p.parse_args()

    types = [t for t in SP.SAND_TYPES + SNOW_TYPES if os.path.exists(f"{args.runs}/intruder_{t}.npz")]
    if not types:
        raise SystemExit(f"no {args.runs}/intruder_<type>.npz files: run scripts/mpm_intruder_test.py first")
    rows = SP.fit_table([f"{args.runs}/intruder_{t}.npz" for t in types])
    for t, r in zip(types, rows):
        if t in SNOW_TYPES:
            r["lift_r"] = 0.0
    for t, r in zip(types, rows):
        print(f"{t:13s}", "  ".join(f"{k}={v:.3g}" for k, v in r.items()))
    with open(args.out, "w") as f:
        json.dump(dict(zip(types, rows)), f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
