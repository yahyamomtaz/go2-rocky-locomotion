"""Review saved mountain trajectories, including FK-based foot clearances.

Reads the recorded height grid rather than regenerating a newer terrain.
Foot gaps are relative to that grid, excluding separate rock colliders.
"""

import argparse
import json
from pathlib import Path

import newton
import newton.utils
import numpy as np
import trimesh
import warp as wp

from go2_rocky.mountain import MountainTerrain


def recorded_surface(data):
    terrain = MountainTerrain.__new__(MountainTerrain)
    terrain.heights = data["mountain_elevation_m"]
    terrain.axis = data["mountain_xy_axis_m"]
    terrain.n = len(terrain.axis)
    terrain.width_m = terrain.length_m = float(terrain.axis[-1] - terrain.axis[0])
    terrain.spacing = terrain.width_m / (terrain.n - 1)
    terrain.route = data["route_world_m"]
    terrain._cache_route()
    return terrain


def support_normal(terrain, position, radius=0.4):
    axis = np.linspace(-radius, radius, 5)
    dx, dy = np.meshgrid(axis, axis)
    design = np.column_stack((dx.ravel(), dy.ravel(), np.ones(dx.size)))
    heights = terrain.height_at(position[0] + dx.ravel(), position[1] + dy.ravel())
    a, b, intercept = np.linalg.lstsq(design, heights, rcond=None)[0]
    normal = np.array((-a, -b, 1.0))
    return normal / np.linalg.norm(normal), float(intercept)


def foot_model(asset_directory):
    # Archived trajectories may have been recorded on another machine.
    asset_directory = Path(asset_directory)
    if not (asset_directory / "usd/go2.usda").is_file():
        asset_directory = Path(newton.utils.download_asset("unitree_go2"))
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(str(Path(asset_directory) / "usd/go2.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()),
        collapse_fixed_joints=False, enable_self_collisions=False,
        joint_ordering="dfs", hide_collision_shapes=True)
    model = builder.finalize(device="cpu")
    bodies, types = model.shape_body.numpy(), model.shape_type.numpy()
    feet = [i for i, body in enumerate(bodies) if body >= 0
            and model.body_label[body].endswith("_foot") and types[i] == int(newton.GeoType.SPHERE)]
    if len(feet) != 4:
        raise ValueError("expected four spherical foot collision shapes")
    return model, feet


def point_gap(terrain, centre, radius):
    # A sphere can touch a neighbouring triangle at a joint between cells.
    col = int((centre[0] + terrain.width_m / 2) / terrain.spacing)
    row = int((centre[1] + terrain.length_m / 2) / terrain.spacing)
    triangles = []
    for j in range(max(0, row - 1), min(terrain.n - 1, row + 2)):
        for i in range(max(0, col - 1), min(terrain.n - 1, col + 2)):
            a = (terrain.axis[i], terrain.axis[j], terrain.heights[j, i])
            b = (terrain.axis[i + 1], terrain.axis[j], terrain.heights[j, i + 1])
            c = (terrain.axis[i], terrain.axis[j + 1], terrain.heights[j + 1, i])
            d = (terrain.axis[i + 1], terrain.axis[j + 1], terrain.heights[j + 1, i + 1])
            triangles.extend(((a, b, d), (a, d, c)))
    triangles = np.asarray(triangles)
    closest = trimesh.triangles.closest_point(triangles, np.tile(centre, (len(triangles), 1)))
    distance = float(np.linalg.norm(closest - centre, axis=1).min())
    if centre[2] < terrain.height_at(*centre[:2]):
        distance = -distance
    return distance - radius


def analyze(path, model, feet):
    with np.load(path) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads(data["metadata_json"].item())
    terrain = recorded_surface(data)
    time, q = data["time_s"], data["joint_q"]
    body_up = np.column_stack((2 * (q[:, 3]*q[:, 5] + q[:, 4]*q[:, 6]),
                               2 * (q[:, 4]*q[:, 5] - q[:, 3]*q[:, 6]),
                               1 - 2 * (q[:, 3]**2 + q[:, 4]**2)))
    raw = np.asarray([terrain.normal_at(*p[:2]) for p in q[:, :3]])
    fitted = np.asarray([support_normal(terrain, p)[0] for p in q[:, :3]])
    raw_grade = np.degrees(np.arccos(np.clip(raw[:, 2], -1, 1)))
    support_grade = np.degrees(np.arccos(np.clip(fitted[:, 2], -1, 1)))
    tilt = np.degrees(np.arccos(np.clip(body_up[:, 2], -1, 1)))
    gaps = np.zeros((len(time), 4))
    state = model.state()
    shape_transform = model.shape_transform.numpy()
    shape_body, sizes = model.shape_body.numpy(), model.shape_scale.numpy()
    for i, coords in enumerate(q):
        state.joint_q.assign(coords)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        poses = state.body_q.numpy()
        for j, shape in enumerate(feet):
            world = wp.transform_multiply(wp.transform(*poses[shape_body[shape]]), wp.transform(*shape_transform[shape]))
            centre = np.asarray(tuple(wp.transform_get_translation(world)))
            gaps[i, j] = point_gap(terrain, centre, sizes[shape, 0])
    moving = time >= meta["settle_s"]
    stable = moving & (tilt < 30)
    if not stable.any():
        stable = moving
    summary = {
        "file": str(path), "status": meta["status"], "progress_m": meta["path_progress_m"],
        "seconds_simulated": float(time[-1]),
        "max_abs_body_lateral_command_m_s": float(np.abs(data["commanded_body_velocity"][moving, 1]).max()),
        "separate_rock_contact_steps": meta["rock_contact_steps"],
        "last_world_tilt_deg": float(tilt[-1]),
        "last_triangle_grade_deg": float(raw_grade[-1]),
        "last_support_grade_deg": float(support_grade[-1]),
        "last_up_dot_triangle_normal": float(np.dot(body_up[-1], raw[-1])),
        "last_up_dot_support_normal": float(np.dot(body_up[-1], fitted[-1])),
        "max_triangle_grade_deg": float(raw_grade[moving].max()),
        "max_support_grade_deg": float(support_grade[moving].max()),
        "min_foot_gap_m": float(gaps[stable].min()),
        "median_lowest_foot_gap_m": float(np.median(gaps[stable].min(axis=1))),
        "fraction_stable_frames_with_a_foot_within_1cm_of_ground": float(np.mean(np.any(np.abs(gaps[stable]) < 0.01, axis=1))),
        "foot_gap_reference": "heightfield only; separate stones excluded",
    }
    if "rock_transform_world" in data and len(data["rock_transform_world"]):
        poses = data["rock_transform_world"]
        source = meta["rocks"]
        foreground = np.array([r.get("coordinate_frame", "trail") != "world_xy" for r in source])
        height = poses[:, 2] - terrain.height_at(poses[:, 0], poses[:, 1])
        summary["foreground_origin_gap_median_m"] = float(np.median(height[foreground]))
        summary["foreground_origin_gap_p95_m"] = float(np.percentile(height[foreground], 95))
        radii = np.asarray([r["bounding_radius_m"] for r in source])
        distance = np.linalg.norm(poses[:, :2] - q[0, :2], axis=1) - radii
        summary["nearest_separate_rock_edge_at_spawn_m"] = float(distance.min())
    arrays = {"time": time, "progress": data["base_position_local_m"][:, 1], "raw_grade": raw_grade,
              "support_grade": support_grade, "tilt": tilt, "foot_gaps": gaps,
              "foot_labels": [model.body_label[int(shape_body[shape])].rsplit("/", 1)[-1] for shape in feet],
              "command": data["commanded_body_velocity"]}
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/trajectory_review"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    with np.load(args.logs[0]) as data:
        meta = json.loads(data["metadata_json"].item())
    model, feet = foot_model(meta["asset_directory"])
    results = [analyze(path, model, feet) for path in args.logs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps([r[0] for r in results], indent=2) + "\n")
    print(json.dumps([r[0] for r in results], indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, len(results), figsize=(6 * len(results), 11), squeeze=False)
        for col, (summary, arrays) in enumerate(results):
            t = arrays["time"]
            axes[0, col].plot(t, arrays["progress"])
            axes[0, col].set_title(Path(summary["file"]).stem)
            axes[0, col].set_ylabel("Route distance (m)")
            axes[1, col].plot(t, arrays["raw_grade"], label="Triangle slope")
            axes[1, col].plot(t, arrays["support_grade"], label="Support-plane slope")
            axes[1, col].plot(t, arrays["tilt"], label="Body tilt")
            axes[1, col].set_ylabel("Angle (degrees)")
            axes[1, col].legend()
            for foot, label in enumerate(arrays["foot_labels"]):
                axes[2, col].plot(t, arrays["foot_gaps"][:, foot] * 100, label=label)
            axes[2, col].legend(fontsize=8, ncol=2)
            axes[2, col].axhline(0, color="black", linewidth=0.7)
            axes[2, col].set_ylabel("Foot / ground gap (cm)")
            for command, label in enumerate(("Forward", "Left", "Yaw")):
                axes[3, col].plot(t, arrays["command"][:, command], label=label)
            axes[3, col].legend(fontsize=8, ncol=3)
            axes[3, col].set_ylabel("Body command (m/s, rad/s)")
            axes[3, col].set_xlabel("Time (s)")
            for ax in axes[:, col]:
                ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
