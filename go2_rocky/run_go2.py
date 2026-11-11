"""Run Newton's pretrained Go2 policy on a mountain or a planar baseline.

Policy interface follows Newton 1.6's Apache-2.0 robot_policy example.
Assets and weights are loaded from Newton's asset cache, not bundled here.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import newton
import newton.utils
import newton.viewer
import numpy as np
import onnx
import warp as wp
import yaml
from warp_nn.runtime import OnnxRuntime

from go2_rocky.drop_sphere import GRAVITY, add_rigid_bed
from go2_rocky.terrain import SlopeTerrain
from go2_rocky.rocks import RockConfig, add_static_rocks, generate_rocks, generate_mountain_rocks
from go2_rocky.controller import PathControllerConfig, path_command
from go2_rocky.mountain import MountainConfig, MountainTerrain, add_mountain_rocks
from go2_rocky.camera import FollowCamera, set_view


@wp.kernel
def observe(q: wp.array[float], qd: wp.array[float], rest: wp.array[float],
            previous: wp.array2d[float], command: wp.vec3, obs: wp.array2d[float]):
    """48 values, in the exact order used by the installed MJW Go2 policy."""
    orientation = wp.quat(q[3], q[4], q[5], q[6])
    linear = wp.quat_rotate_inv(orientation, wp.vec3(qd[0], qd[1], qd[2]))
    angular = wp.quat_rotate_inv(orientation, wp.vec3(qd[3], qd[4], qd[5]))
    gravity = wp.quat_rotate_inv(orientation, wp.vec3(0.0, 0.0, -1.0))
    for i in range(3):
        obs[0, i] = linear[i]
        obs[0, 3 + i] = angular[i]
        obs[0, 6 + i] = gravity[i]
        obs[0, 9 + i] = command[i]
    for j in range(12):
        obs[0, 12 + j] = q[7 + j] - rest[j]
        obs[0, 24 + j] = qd[6 + j]
        obs[0, 36 + j] = previous[0, j]


@wp.kernel
def joint_targets(action: wp.array2d[float], rest: wp.array[float],
                  scale: float, target: wp.array[float]):
    j = wp.tid()
    target[7 + j] = rest[j] + scale * action[0, j]


class Go2Run:
    physics_dt = 0.005  # Match the upstream policy example: 200 Hz physics.
    decimation = 4      # One action every 0.02 s: 50 Hz policy.
    policy_dt = physics_dt * decimation
    width_m = 4.0
    length_m = 12.0

    def __init__(self, terrain, speed=0.3, settle=1.0, friction=0.75,
                 device="cuda:0", policy_path=None, rock_config=None, controller="open-loop",
                 goal_distance=None, rock_overrides=None):
        self.terrain, self.speed, self.settle = terrain, speed, settle
        self.is_mountain = isinstance(terrain, MountainTerrain)
        if self.is_mountain:
            self.width_m, self.length_m = terrain.width_m, terrain.length_m
        if controller not in ("open-loop", "path"):
            raise ValueError("controller must be open-loop or path")
        self.controller = controller
        if goal_distance is not None and (not math.isfinite(goal_distance) or goal_distance <= 0):
            raise ValueError("goal_distance must be finite and positive, or None")
        self.goal_distance = goal_distance
        self.rock_config = rock_config
        if rock_config and not self.is_mountain and (rock_config.width_m != self.width_m or rock_config.length_m != self.length_m):
            raise ValueError("rock layout dimensions must match the supporting bed")
        self.rocks = (tuple(rock_overrides) if rock_overrides is not None else generate_mountain_rocks(terrain, rock_config)
                      if self.is_mountain and rock_config and rock_config.distribution == "mountain"
                      else generate_rocks(rock_config) if rock_config else ())
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("this runner requires CUDA for MuJoCo Warp")
        self.asset_dir = Path(newton.utils.download_asset("unitree_go2"))
        config_path = self.asset_dir / "rl_policies/go2.yaml"
        self.config = yaml.safe_load(config_path.read_text())
        if self.config["num_dofs"] != 12:
            raise ValueError("expected the 12-joint Go2 policy configuration")
        self.policy_path = Path(policy_path) if policy_path else self.asset_dir / "rl_policies/mjw_go2.onnx"
        if self.policy_path.suffix != ".onnx":
            raise ValueError("this runner supports ONNX policies; .pt is not interchangeable")
        network = onnx.load(str(self.policy_path))
        for tensor, size in ((network.graph.input[0], 48), (network.graph.output[0], 12)):
            if tensor.type.tensor_type.shape.dim[-1].dim_value != size:
                raise ValueError(f"policy must have 48 observations and 12 actions: {tensor.name}")
        self.policy = OnnxRuntime(str(self.policy_path), device=self.device)
        self.input_name = self.policy.input_names[0]
        self.output_name = self.policy.output_names[0]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1, limit_ke=100.0, limit_kd=1.0
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = friction
        builder.rigid_gap = 0.0
        builder.add_usd(
            str(self.asset_dir / "usd/go2.usda"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
            collapse_fixed_joints=False, enable_self_collisions=False,
            joint_ordering="dfs", hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")
        actual_order = [
            label.rsplit("/", 1)[-1] for label, kind in zip(builder.joint_label, builder.joint_type)
            if kind == newton.JointType.REVOLUTE
        ]
        if actual_order != self.config["mjw_joint_names"]:
            raise ValueError(f"Go2 joint ordering does not match the policy: {actual_order}")
        self.rock_placements = np.empty((0, 7))
        if self.is_mountain:
            terrain.add_collision(builder, textured=True)
            self.rock_shape_ids, self.rock_placements = add_mountain_rocks(
                builder, terrain, self.rocks, rock_config.length_m if rock_config else 0
            )
        else:
            add_rigid_bed(builder, terrain, self.width_m, self.length_m,
                          color=(0.30, 0.28, 0.25) if self.rocks else (0.35, 0.5, 0.35))
            self.rock_shape_ids = add_static_rocks(builder, terrain, self.rocks)

        # Body +X is forward. Yaw +90 degrees maps it onto terrain-local +v.
        if self.is_mountain:
            frame = terrain.frame_at(terrain.route[0])
            body_rotation = np.column_stack((frame[:, 1], -frame[:, 0], frame[:, 2]))
            builder.joint_q[:3] = terrain.route[0] + frame[:, 2] * 0.5
            builder.joint_q[3:7] = list(wp.quat_from_matrix(wp.mat33(*body_rotation.ravel())))
        else:
            bed_rotation = wp.quat_from_matrix(wp.mat33(*(
                value for row in terrain.rotation_matrix for value in row
            )))
            yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2)
            builder.joint_q[:3] = terrain.local_to_world((0.0, -4.5, 0.5))
            builder.joint_q[3:7] = list(wp.mul(bed_rotation, yaw))
        builder.joint_q[7:] = self.config["mjw_joint_pos"]
        for j in range(12):
            builder.joint_target_ke[6 + j] = self.config["mjw_joint_stiffness"][j]
            builder.joint_target_kd[6 + j] = self.config["mjw_joint_damping"][j]
            builder.joint_armature[6 + j] = self.config["mjw_joint_armature"][j]
            builder.joint_target_mode[6 + j] = int(newton.JointTargetMode.POSITION)
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity(GRAVITY)

        # Measure the actual spherical foot collision geometry with FK, then
        # translate the whole robot so the lowest foot is 2 cm above the bed.
        probe = self.model.state()
        newton.eval_fk(self.model, probe.joint_q, probe.joint_qd, probe)
        body_q = probe.body_q.numpy()
        shape_q = self.model.shape_transform.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_type = self.model.shape_type.numpy()
        shape_scale = self.model.shape_scale.numpy()
        self.foot_shapes = [
            i for i, body in enumerate(shape_body)
            if body >= 0 and self.model.body_label[body].endswith("_foot")
            and shape_type[i] == int(newton.GeoType.SPHERE)
        ]
        if len(self.foot_shapes) != 4:
            raise ValueError("expected four spherical foot colliders; inspect the Go2 asset")
        self.foot_labels = [self.model.body_label[int(shape_body[i])].rsplit("/", 1)[-1]
                            for i in self.foot_shapes]
        clearances = []
        vertical_shifts = []
        for i in self.foot_shapes:
            world = wp.transform_multiply(wp.transform(*body_q[shape_body[i]]), wp.transform(*shape_q[i]))
            centre = tuple(wp.transform_get_translation(world))
            if self.is_mountain:
                normal_z = terrain.normal_at(*centre[:2])[2]
                clearance = (centre[2] - terrain.height_at(*centre[:2])) * normal_z - shape_scale[i, 0]
                vertical_shifts.append((0.02 - clearance) / normal_z)
            else:
                clearances.append(terrain.world_to_local(centre)[2] - shape_scale[i, 0])
        initial = self.model.joint_q.numpy()
        if self.is_mountain:
            initial[2] += max(vertical_shifts)
        else:
            initial[:3] += np.asarray(terrain.normal) * (0.02 - min(clearances))
        self.model.joint_q.assign(initial)
        self.initial_foot_clearance_m = 0.02

        self.state = self.model.state()
        self.next_state = self.model.state()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            nconmax=128 if self.rocks else 64, njmax=512 if self.rocks else 256,
        )
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        self.rock_contact_steps = 0
        self.rock_contact_count = 0
        self.foot_contact_counts = np.zeros(4, dtype=int)
        self.control = self.model.control()
        self.control.joint_target_q = wp.zeros(19, dtype=wp.float32, device=self.device)
        self.rest = wp.array(self.config["mjw_joint_pos"], dtype=wp.float32, device=self.device)
        self.previous = wp.zeros((1, 12), dtype=wp.float32, device=self.device)
        self.obs = wp.zeros((1, 48), dtype=wp.float32, device=self.device)
        self.time = 0.0
        self.steps = 0
        self.status = "running"
        self.history = []
        self.move_start = None
        self.record((0.0, 0.0, 0.0))

    def record(self, command):
        q = self.state.joint_q.numpy().copy()
        qd = self.state.joint_qd.numpy().copy()
        local = np.asarray(self.terrain.path_state(q[:3]))
        self.history.append((self.time, q, qd, local, command[0], self.rock_contact_count, tuple(command), self.foot_contact_counts.copy()))
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            self.status = "numerical_instability"
        elif local[2] < 0.18 or float(wp.dot(
            wp.quat_rotate(wp.quat(*q[3:7]), wp.vec3(0.0, 0.0, 1.0)),
            wp.vec3(*self.terrain.normal_at(*q[:2])),
        )) < 0.4:
            self.status = "fell_during_settling" if self.time <= self.settle else "fell"
        elif (self.is_mountain and (np.max(np.abs(q[:2])) > self.width_m / 2 - 0.6 or abs(local[0]) > 3)) or (
            not self.is_mountain and (abs(local[0]) > self.width_m / 2 - 0.6 or abs(local[1]) > self.length_m / 2 - 0.6)
        ):
            self.status = "out_of_bounds"
        elif self.goal_distance is not None and self.move_start is not None and self.terrain.progress(
            self.move_start, q[:3]
        ) >= self.goal_distance:
            self.status = "goal_reached"

    def step(self):
        walking = self.time >= self.settle - 1e-9
        command = (self.speed, 0.0, 0.0) if walking else (0.0, 0.0, 0.0)
        if walking and self.controller == "path":
            command = path_command(self.terrain, self.history[-1][1], self.history[-1][2], self.speed)
        if self.move_start is None and self.time >= self.settle - 1e-9:
            self.move_start = self.state.joint_q.numpy()[:3].copy()
        wp.launch(observe, dim=1, inputs=[
            self.state.joint_q, self.state.joint_qd, self.rest, self.previous,
            wp.vec3(*command), self.obs,
        ], device=self.device)
        action = self.policy({self.input_name: self.obs})[self.output_name]
        if not np.isfinite(action.numpy()).all():
            self.status = "numerical_instability"
            return
        wp.launch(joint_targets, dim=12, inputs=[
            action, self.rest, float(self.config["action_scale"]), self.control.joint_target_q,
        ], device=self.device)
        wp.copy(self.previous, action)
        for _ in range(self.decimation):
            self.state.clear_forces()
            self.solver.step(self.state, self.next_state, self.control, None, self.physics_dt)
            self.state, self.next_state = self.next_state, self.state
        if self.contacts is not None:
            self.solver.update_contacts(self.contacts, self.state)
            count = int(self.contacts.rigid_contact_count.numpy()[0])
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
            self.rock_contact_count = int(np.count_nonzero(
                np.isin(shape0, self.rock_shape_ids) | np.isin(shape1, self.rock_shape_ids)
            ))
            self.foot_contact_counts = np.asarray([
                np.count_nonzero((shape0 == shape) | (shape1 == shape)) for shape in self.foot_shapes
            ])
            self.rock_contact_steps += int(self.rock_contact_count > 0)
        self.steps += 1
        self.time = self.steps * self.policy_dt
        self.record(command)

    def results(self):
        start = self.move_start if self.move_start is not None else self.history[0][1][:3]
        end = self.history[-1][1][:3]
        progress = float(self.terrain.progress(start, end))
        rise = float(end[2] - start[2])
        return {
            "status": self.status, "path_progress_m": progress,
            "controller": self.controller,
            "goal_reached": self.status == "goal_reached" if self.goal_distance is not None else None,
            "completion_time_s": self.time - self.settle if self.status == "goal_reached" else None,
            "max_abs_cross_track_m": float(max(abs(h[3][0]) for h in self.history)),
            "terrain": "mountain" if self.is_mountain else "rocky" if self.rocks else "smooth", "rock_count": len(self.rocks),
            "embedded_rock_cells": self.terrain.surface.cell_count if self.is_mountain and self.terrain.config.surface_rockiness else 0,
            "rock_contact_steps": self.rock_contact_steps,
            "base_height_gain_m": rise,
            "terrain_height_gain_m": float(
                self.terrain.height_at(*end[:2]) - self.terrain.height_at(*start[:2])
            ),
            "seconds_simulated": self.time,
        }

    def save(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        mountain_data = {
            "mountain_elevation_m": self.terrain.heights,
            "mountain_xy_axis_m": self.terrain.axis,
            "route_world_m": self.terrain.route,
            "route_distance_m": self.terrain.route_distance_m,
            "rock_transform_world": self.rock_placements,
            "surface_relief_m": self.terrain.surface_relief_m,
        } if self.is_mountain else {}
        np.savez_compressed(
            path, time_s=np.asarray([h[0] for h in self.history]),
            joint_q=np.asarray([h[1] for h in self.history]),
            joint_qd=np.asarray([h[2] for h in self.history]),
            base_position_local_m=np.asarray([h[3] for h in self.history]),
            base_position_world_m=np.asarray([h[1][:3] for h in self.history]),
            commanded_forward_speed_m_s=np.asarray([h[4] for h in self.history]),
            commanded_body_velocity=np.asarray([h[6] for h in self.history]),
            rock_contact_count=np.asarray([h[5] for h in self.history]),
            foot_contact_count=np.asarray([h[7] for h in self.history]),
            metadata_json=json.dumps(metadata),
            **mountain_data,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=("smooth", "rocky", "mountain"), default="mountain")
    parser.add_argument("--slope-deg", type=float, default=15.0,
                        help="Bed inclination, or target mountain trail grade; local mountain slopes vary (degrees)")
    parser.add_argument("--roughness", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mountain-size", type=float, default=None, help="Mountain width and length in metres (default 80)")
    parser.add_argument("--mountain-height", type=float, default=None, help="Main peak height scale in metres (default 26)")
    parser.add_argument("--terrain-resolution", type=float, default=None, help="Heightfield grid spacing in metres (default 0.2)")
    parser.add_argument("--surface-rockiness", type=float, default=None,
                        help="Mountain-wide bedrock relief multiplier, 0 to 3 (default 1)")
    parser.add_argument("--rock-count", type=int, default=None,
                        help="Separate collision rocks; mountain defaults: low 800, medium 1400, high 2000")
    parser.add_argument("--friction-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Rock friction range; --friction controls the supporting bed")
    parser.add_argument("--direction", choices=("uphill", "downhill", "cross"), default="uphill")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Path speed command in path mode, body-forward command in open-loop mode (m/s)")
    parser.add_argument("--controller", choices=("open-loop", "path"), default="path",
                        help="path uses simulator pose to correct heading and lateral drift")
    parser.add_argument("--goal-distance", type=float, default=None,
                        help="Stop at this path progress after settling; 0 disables the goal (metres)")
    parser.add_argument("--seconds", type=float, default=60.0, help="Maximum walking time, after settling")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.75)
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--camera", choices=("follow", "wide", "free"), default=None,
                        help="follow keeps the robot centred; free preserves mouse camera controls")
    parser.add_argument("--camera-distance", type=float, default=FollowCamera.DEFAULT_DISTANCE,
                        help="Follow camera horizontal distance in metres (default 3)")
    parser.add_argument("--camera-height", type=float, default=FollowCamera.DEFAULT_HEIGHT,
                        help="Follow camera height above the robot, before obstacle clearance (default 1.8)")
    parser.add_argument("--headless", action="store_true", help="Render OpenGL offscreen; requires --viewer gl")
    parser.add_argument("--screenshot", type=Path, help="Save the last OpenGL frame as .png")
    parser.add_argument("--video", type=Path, help="Record the OpenGL view to .mp4 (install the video extra)")
    parser.add_argument("--video-fps", type=int, default=30, help="MP4 playback rate, 1–50 (default 30)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", type=Path, help="Compatible ONNX checkpoint; default: Newton's MJW Go2 policy")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--check", action="store_true", help="Require an upright run and at least 0.5 m forward progress")
    args = parser.parse_args()
    if args.goal_distance is None:
        args.goal_distance = 5.0 if args.terrain == "mountain" else 2.0
    if not math.isfinite(args.goal_distance) or args.goal_distance < 0:
        parser.error("--goal-distance must be finite and nonnegative (0 disables the goal)")
    goal_distance = args.goal_distance or None
    if args.terrain != "mountain" and any(v is not None for v in (
        args.mountain_size, args.mountain_height, args.terrain_resolution, args.surface_rockiness
    )):
        parser.error("mountain geometry options require --terrain mountain")
    if args.terrain == "mountain" and args.direction != "uphill":
        parser.error("mountain mode currently supports its uphill trail only")
    if args.terrain == "mountain" and args.rock_count == 0 and args.friction_range is not None:
        parser.error("rock friction range requires separate rocks; omit --rock-count 0")
    if args.viewer != "gl" and (args.headless or args.screenshot or args.video):
        parser.error("--headless, --screenshot, and --video require --viewer gl")
    if args.video and args.video.suffix.lower() != ".mp4":
        parser.error("--video must end in .mp4")
    if not 1 <= args.video_fps <= 50:
        parser.error("--video-fps must be in [1, 50]")
    if any(not math.isfinite(v) or v <= 0 for v in (args.camera_distance, args.camera_height)):
        parser.error("camera distance and height must be finite and positive")
    if args.screenshot and args.screenshot.suffix.lower() != ".png":
        parser.error("--screenshot must end in .png")
    if args.terrain == "smooth" and any(v is not None for v in (
        args.roughness, args.seed, args.rock_count, args.friction_range
    )):
        parser.error("rock options require --terrain rocky or mountain")
    for name in ("speed", "seconds", "settle", "friction"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ("speed", "seconds") and value == 0):
            parser.error(f"--{name} must be finite and {'positive' if name in ('speed', 'seconds') else 'nonnegative'}")
    if args.log and args.log.suffix != ".npz":
        parser.error("--log must end in .npz")
    try:
        terrain = MountainTerrain(MountainConfig(
            seed=args.seed if args.seed is not None else 1,
            size_m=args.mountain_size if args.mountain_size is not None else 80.0,
            height_m=args.mountain_height if args.mountain_height is not None else 26.0,
            resolution_m=args.terrain_resolution if args.terrain_resolution is not None else 0.2,
            trail_grade_deg=args.slope_deg,
            surface_rockiness=args.surface_rockiness if args.surface_rockiness is not None else 1.0,
            roughness=args.roughness or "medium",
        )) if args.terrain == "mountain" else SlopeTerrain(args.slope_deg, args.direction)
        if args.terrain == "mountain" and goal_distance is not None and goal_distance > terrain.route_distance_m[-1] - 1:
            raise ValueError("goal distance exceeds the generated mountain trail")
        rock_config = RockConfig(
            roughness=args.roughness or "medium",
            seed=args.seed if args.seed is not None else 1,
            rock_count=args.rock_count, width_m=terrain.width_m if args.terrain == "mountain" else Go2Run.width_m,
            length_m=terrain.length_m if args.terrain == "mountain" else Go2Run.length_m,
            distribution="mountain" if args.terrain == "mountain" else "trail",
            friction_range=tuple(args.friction_range) if args.friction_range else (args.friction, args.friction),
            start_clearance_m=0.9 if args.terrain == "mountain" else 2.4,
        ) if args.terrain in ("rocky", "mountain") and not (args.terrain == "mountain" and args.rock_count == 0) else None
    except ValueError as error:
        parser.error(str(error))
    wp.init()
    with wp.ScopedDevice(args.device):
        sim = Go2Run(terrain, args.speed, args.settle, args.friction, args.device, args.policy,
                     rock_config=rock_config, controller=args.controller, goal_distance=goal_distance)
        print(f"Terrain: {args.terrain}; {'trail grade target' if sim.is_mountain else 'slope'}: {terrain.slope_deg:g} deg; fixed rocks: {len(sim.rocks)}")
        frames = math.ceil((args.seconds + args.settle) / sim.policy_dt)
        viewer = (
            newton.viewer.ViewerGL(num_frames=frames+1, headless=args.headless, width=1280, height=720)
            if args.viewer == "gl" else newton.viewer.ViewerNull(num_frames=frames)
        )
        recorder = None
        camera_mode = args.camera or ("follow" if sim.is_mountain else "free")
        try:
            viewer.set_model(sim.model)
            if sim.is_mountain and args.viewer == "gl":
                terrain.log_visual(viewer, sim.device)
            follow = FollowCamera.from_simulation(sim, distance=args.camera_distance,
                                                  height=args.camera_height) if args.viewer == "gl" else None
            if follow:
                set_view(viewer, sim, camera_mode, follow, 0, initial=True)
                viewer.begin_frame(0)
                viewer.log_state(sim.state)
                viewer.end_frame()
            if args.video:
                from go2_rocky.video import VideoRecorder
                recorder = VideoRecorder(args.video, args.video_fps)
                recorder.capture(0, lambda: viewer.get_frame().numpy())
            while sim.steps < frames and sim.status == "running" and viewer.is_running():
                if not viewer.is_paused():
                    sim.step()
                if follow:
                    set_view(viewer, sim, camera_mode, follow, sim.time)
                viewer.begin_frame(sim.time)
                viewer.log_state(sim.state)
                viewer.end_frame()
                if recorder:
                    recorder.capture(sim.time, lambda: viewer.get_frame().numpy())
            if sim.status == "running":
                sim.status = "time_limit" if sim.steps == frames else "viewer_closed"
            if args.screenshot:
                from PIL import Image
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
                print(f"Saved {args.screenshot}")
        finally:
            try:
                if recorder:
                    recorder.close()
                    print(f"Saved {args.video} ({recorder.frames} frames at {recorder.fps} fps)")
            finally:
                viewer.close()
        result = sim.results()
        metadata = {
            **result, "slope_deg": terrain.slope_deg, "direction": terrain.direction,
            "camera": camera_mode, "video_path": str(args.video) if args.video else None,
            "camera_distance_m": args.camera_distance, "camera_height_m": args.camera_height,
            "video_fps": args.video_fps if args.video else None,
            "speed_m_s": args.speed, "settle_s": args.settle, "walking_seconds_requested": args.seconds,
            "speed_command_frame": "terrain_path" if args.controller == "path" else "body_forward",
            "commanded_body_velocity_columns": ["forward_m_s", "left_m_s", "yaw_rad_s"],
            "gravity": GRAVITY, "backend": "mujoco-warp", "friction": args.friction,
            "physics_dt_s": sim.physics_dt, "policy_dt_s": sim.policy_dt,
            "bed_size_m": None if sim.is_mountain else [sim.width_m, sim.length_m, 0.2],
            "terrain_extent_m": [sim.width_m, sim.length_m],
            "heightfield_range_m": [float(terrain.heights.min()), float(terrain.heights.max())] if sim.is_mountain else None,
            "path_coordinate_columns": ["cross_track_m", "route_progress_m", "normal_clearance_m"],
            "mountain_config": asdict(terrain.config) if sim.is_mountain else None,
            "elevation_sha256": hashlib.sha256(terrain.heights.tobytes()).hexdigest() if sim.is_mountain else None,
            "initial_foot_clearance_m": sim.initial_foot_clearance_m,
            "rock_config": asdict(rock_config) if rock_config else None,
            "rocks": [asdict(rock) for rock in sim.rocks],
            "terrain_height_gain_definition": "heightfield elevation change at base XY; excludes individual rocks" if sim.is_mountain
                else "supporting bed elevation change; excludes individual rock heights",
            "controller_inputs": "simulator base pose, angular velocity, reference route and local surface frame"
                if args.controller == "path" else "constant body-forward velocity command",
            "controller_parameters": asdict(PathControllerConfig()) if args.controller == "path" else {},
            "controller_revision": "body_command_bounds_v2",
            "foot_contact_count_definition": "native detected contacts per spherical foot collider, not a force measurement",
            "foot_contact_count_columns": sim.foot_labels,
            "goal_distance_m": goal_distance,
            "asset_directory": str(sim.asset_dir), "policy_path": str(sim.policy_path),
            "policy_sha256": hashlib.sha256(sim.policy_path.read_bytes()).hexdigest(),
            "versions": {name: version(name) for name in ("newton", "warp-lang", "mujoco", "mujoco-warp", "warp-nn", "opensimplex")},
        }
        if args.log:
            sim.save(args.log, metadata)
            print(f"Saved {args.log}")
        print(json.dumps(result, indent=2))
        if args.check:
            if goal_distance is not None and sim.status != "goal_reached":
                raise AssertionError(f"Go2 did not reach the {goal_distance:g} m traversal goal: {sim.status}")
            if sim.status not in ("time_limit", "goal_reached") or result["path_progress_m"] < min(0.5, goal_distance or 0.5):
                raise AssertionError("Go2 failed to complete the run upright with 0.5 m forward progress")
            if terrain.direction == "uphill" and terrain.slope_deg > 0 and result["base_height_gain_m"] <= 0:
                raise AssertionError("Go2 did not gain height while walking uphill")
            if sim.rocks and sim.rock_contact_steps == 0:
                raise AssertionError("Go2 did not contact any rocks; run longer to enter the rocky area")
            np.testing.assert_allclose(sim.model.gravity.numpy(), [GRAVITY], atol=1e-6)
            print("PASS: Go2 walked forward; uphill runs also gained height")


if __name__ == "__main__":
    main()
