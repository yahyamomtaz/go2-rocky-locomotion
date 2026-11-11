"""Primitive-contact checkpoint: a sphere falling onto a finite rigid slope.

Run with ``python -m go2_rocky.drop_sphere --slope-deg 5 --viewer gl``.
No robot, policy, sand, or additional floor is involved in this experiment.
"""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import newton
import newton.viewer
import numpy as np
import warp as wp

from go2_rocky.terrain import SlopeTerrain


GRAVITY = (0.0, 0.0, -9.81)


def add_rigid_bed(builder, terrain, width_m=4.0, length_m=8.0, thickness_m=0.2,
                  color=(0.35, 0.5, 0.35)):
    """Create a static collision box with its TOP face at terrain-local w=0."""
    if any(not math.isfinite(x) or x <= 0 for x in (width_m, length_m, thickness_m)):
        raise ValueError("bed dimensions must be finite and positive")

    # Newton boxes are centred at their origin and take HALF extents.
    centre = terrain.local_to_world((0.0, 0.0, -thickness_m / 2))
    rotation = wp.quat_from_matrix(wp.mat33(*(
        value for row in terrain.rotation_matrix for value in row
    )))
    return builder.add_shape_box(
        body=-1,  # Attach to the world: the bed cannot move.
        xform=wp.transform(wp.vec3(*centre), rotation),
        hx=width_m / 2,
        hy=length_m / 2,
        hz=thickness_m / 2,
        color=color,
        label="rigid_bed",
    )


class SphereDrop:
    """Small, deliberately uncaptured simulation loop for learning/debugging."""

    dt = 1.0 / 480.0
    substeps = 8  # Render at 60 Hz; physics runs at 480 Hz.
    radius = 0.1
    width_m = 4.0
    length_m = 8.0
    thickness_m = 0.2

    def __init__(self, terrain, device="cuda:0", friction=0.75):
        if not math.isfinite(friction) or friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        self.terrain = terrain
        self.friction = friction
        self.device = wp.get_device(device)
        # CPU is an explicit diagnostic choice, not a silent GPU fallback.
        self.backend = "mujoco-cpu" if self.device.is_cpu else "mujoco-warp"
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = friction
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        self.bed_shape = add_rigid_bed(
            builder, terrain, self.width_m, self.length_m, self.thickness_m
        )

        # Start 0.5 m clear of the surface, measured along its normal.
        spawn = terrain.local_to_world((0.0, 0.0, self.radius + 0.5))
        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(*spawn), wp.quat_identity()), label="sphere"
        )
        builder.add_shape_sphere(
            self.sphere_body, radius=self.radius, color=(0.9, 0.35, 0.15)
        )
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity(GRAVITY)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.device.is_cpu,
            use_mujoco_contacts=True,
            solver="newton",
            nconmax=16,
            njmax=64,
        )
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        # Newton 1.6 only exports its Contacts buffer on the Warp backend.
        self.contacts = (
            None if self.device.is_cpu
            else newton.Contacts(self.solver.get_max_contact_count(), 0)
        )
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.time = [0.0]
        self.positions = [self.state.body_q.numpy()[self.sphere_body, :3].copy()]
        self.contact_counts = [0]

    def step(self):
        self.state.clear_forces()
        # MuJoCo performs collision detection against the actual bed shape.
        self.solver.step(self.state, self.next_state, self.control, None, self.dt)
        self.state, self.next_state = self.next_state, self.state
        if self.device.is_cpu:
            contact_count = int(self.solver.mj_data.ncon)
        else:
            self.solver.update_contacts(self.contacts, self.state)
            contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.time.append(len(self.time) * self.dt)
        self.positions.append(self.state.body_q.numpy()[self.sphere_body, :3].copy())
        self.contact_counts.append(contact_count)

    def arrays(self):
        positions = np.asarray(self.positions)
        local = np.asarray([self.terrain.world_to_local(p) for p in positions])
        return {
            "time_s": np.asarray(self.time),
            "sphere_position_world_m": positions,
            "sphere_position_local_m": local,
            "clearance_m": local[:, 2] - self.radius,
            "contact_count": np.asarray(self.contact_counts),
        }

    def check(self):
        """Reject a missing contact, excessive penetration, or leaving the bed."""
        data = self.arrays()
        if not all(np.isfinite(a).all() for a in data.values()):
            raise AssertionError("trajectory contains NaN/Inf")
        np.testing.assert_allclose(self.model.gravity.numpy().reshape(-1, 3), [GRAVITY], atol=1e-6)
        if not np.any(data["contact_count"] > 0):
            raise AssertionError("no sphere/bed contact detected; try --seconds 2")
        if data["clearance_m"].min() < -0.01:
            raise AssertionError("sphere penetrated the bed by more than 1 cm")
        tail_steps = max(1, round(0.25 / self.dt))
        if np.max(np.abs(data["clearance_m"][-tail_steps:])) > 0.01:
            raise AssertionError("sphere did not remain supported during the final 0.25 s")
        local = data["sphere_position_local_m"]
        if np.any(np.abs(local[:, 0]) + self.radius >= self.width_m / 2) or np.any(
            np.abs(local[:, 1]) + self.radius >= self.length_m / 2
        ):
            raise AssertionError("sphere reached a bed edge; use a shorter run or smaller slope")
        # Before impact, acceleration must be vertical, even on a slope.
        early = data["time_s"] <= 0.15
        t = data["time_s"][early]
        expected = np.tile(data["sphere_position_world_m"][0], (len(t), 1))
        # Semi-implicit Euler position for initially zero velocity.
        expected[:, 2] += 0.5 * GRAVITY[2] * t * (t + self.dt)
        np.testing.assert_allclose(data["sphere_position_world_m"][early], expected, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slope-deg", type=float, default=5.0)
    parser.add_argument("--direction", choices=("uphill", "downhill", "cross"), default="uphill")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=0.75)
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 uses MuJoCo Warp; cpu uses MuJoCo C")
    parser.add_argument("--log", type=Path, help="Optional trajectory .npz path")
    parser.add_argument("--check", action="store_true", help="Validate contacts, support, and vertical gravity")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.log and args.log.suffix != ".npz":
        parser.error("--log must end in .npz")
    try:
        terrain = SlopeTerrain(args.slope_deg, args.direction)
        if not math.isfinite(args.friction) or args.friction < 0:
            raise ValueError("friction must be finite and nonnegative")
    except ValueError as error:
        parser.error(str(error))

    wp.init()
    with wp.ScopedDevice(args.device):
        sim = SphereDrop(terrain, device=args.device, friction=args.friction)
        frames = math.ceil(args.seconds * 60)
        viewer_type = newton.viewer.ViewerGL if args.viewer == "gl" else newton.viewer.ViewerNull
        viewer = viewer_type(num_frames=frames)
        try:
            viewer.set_model(sim.model)
            viewer.set_camera(pos=wp.vec3(5.0, -7.0, 4.0), pitch=-25.0, yaw=125.0)
            completed_frames = 0
            while completed_frames < frames and viewer.is_running():
                if not viewer.is_paused():
                    for _ in range(sim.substeps):
                        sim.step()
                    completed_frames += 1
                viewer.begin_frame(sim.time[-1])
                viewer.log_state(sim.state)
                viewer.end_frame()
        finally:
            viewer.close()

        metadata = {
            "experiment": "sphere_drop", "slope_deg": terrain.slope_deg,
            "direction": terrain.direction, "gravity": GRAVITY,
            "backend": sim.backend, "device": str(sim.device),
            "dt_s": sim.dt, "seconds_requested": args.seconds,
            "seconds_simulated": sim.time[-1], "friction": args.friction,
            "bed_size_m": [sim.width_m, sim.length_m, sim.thickness_m],
            "sphere_radius_m": sim.radius,
            "versions": {name: version(name) for name in ("newton", "warp-lang", "mujoco", "mujoco-warp")},
        }
        data = sim.arrays()
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.log, **data, metadata_json=json.dumps(metadata))
            print(f"Saved {args.log}")
        print(f"Backend: {sim.backend}; gravity: {GRAVITY}")
        print(f"Minimum clearance: {data['clearance_m'].min():.6f} m")
        print(f"Final terrain coordinates: {data['sphere_position_local_m'][-1]}")
        if args.check:
            sim.check()
            print("PASS: vertical gravity, collision contact, and sphere support")


if __name__ == "__main__":
    main()
