"""Unitree Go2 walking on sand (two-way coupled, converged MPM), or on rigid ground as the baseline.

    uv run python scripts/go2_sand.py --seconds 6 --speed 0.6 --log runs/go2_sand.npz
    uv run python scripts/go2_sand.py --rigid --seconds 6 --speed 0.6 --log runs/go2_rigid.npz

The Go2 and its walking policy are Newton's (newton-assets unitree_go2: usd/go2.usda, rl_policies/mjw_go2.onnx,
trained with MuJoCo Warp, 5 ms physics, 50 Hz policy). On sand, one shared model holds the robot (MuJoCo Warp, no
rigid contacts) and a sand bed (implicit MPM, 500 Gauss-Seidel iterations, checked against Terzaghi bearing theory) whose floor
and walls touch only sand, so the robot is carried by the sand alone. Newton's SolverCoupledProxy exposes the robot's
links to the sand as proxy colliders with their effective (articulated) masses and returns the sand's impulses as
lagged wrenches. With --rigid the same robot walks on a rigid plane with the policy's training settings.
Logged at 50 Hz: base pose and velocity, foot centres and velocities, per-foot sand force, joint state and targets.
"""
import os
import time

import numpy as np
import warp as wp
import yaml
from newton.solvers.experimental.coupled import SolverCoupledProxy
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.examples.robot.example_robot_policy import _build_joint_target_q_kernel, _compute_obs_kernel
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
SNOW = {"settled": (250.0, 0.36, 1.0e5, 6.0, 1.0, 3000.0),   # fits Bekker US/Swedish snow, log-rms 0.16/0.14
        "powder": (150.0, 0.36, 5.0e4, 6.0, 1.0, 1000.0),    # extrapolation: lighter, weaker, less cohesive
        "packed": (400.0, 0.36, 2.0e5, 6.0, 1.0, 8000.0)}    # extrapolation: denser, stronger, more cohesive
# (initial Jp 1.0: the calibration's --jp0 was written to the wrong state and never took effect, so 1.0 is what was
# actually calibrated and used in every snow run)
FOOT_RADIUS = 0.022


@wp.kernel
def _body_force_kernel(impulse: wp.array[wp.vec3], cid: wp.array[int], collider_body: wp.array[int],
                       body_slot: wp.array[int], inv_dt: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    c = cid[i]
    if c < 0:
        return
    b = collider_body[c]
    if b < 0:
        return
    s = body_slot[b]
    if s >= 0:
        wp.atomic_add(out, s, impulse[i] * inv_dt)


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if args.snow:   # snow presets: (density, friction, compaction strength, hardening, initial Jp, cohesion knob)
            d, mu, yp, hard, jp0, coh = SNOW[args.snow]
            args.density, args.friction, args.yield_pressure, args.hardening, args.jp0, args.cohesion = (
                d, mu, yp, hard, jp0, coh)
        self.policy_dt = 0.02
        self.dt = args.dt
        self.steps_per_policy = max(1, round(self.policy_dt / self.dt))
        self.sim_time = 0.0
        self.device = wp.get_device()
        depth = 0.0 if args.rigid else args.sand_depth
        self.surface = depth

        asset = newton.utils.download_asset("unitree_go2")
        with open(asset / "rl_policies" / "go2.yaml", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        nd = self.cfg["num_dofs"]

        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(b)
        SolverImplicitMPM.register_custom_attributes(b)
        b.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
        b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.kf = 5.0e4, 5.0e2, 1.0e3
        b.default_shape_cfg.mu = 0.75
        b.rigid_gap = 0.0
        b.add_usd(str(asset / "usd" / "go2.usda"), xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
                  collapse_fixed_joints=False, enable_self_collisions=False, joint_ordering="dfs",
                  hide_collision_shapes=True)
        b.approximate_meshes("convex_hull")
        self.robot_bodies = list(range(b.body_count))
        self.robot_joints = list(range(b.joint_count))
        self.feet = [next(i for i in range(b.body_count) if b.body_label[i].endswith(f)) for f in FEET]
        b.joint_q[3:7] = [0.0, 0.0, 0.7071068, 0.7071068]   # facing +y, the long axis of the sand bed
        b.joint_q[7:] = self.cfg["mjw_joint_pos"]
        for i in range(nd):
            b.joint_target_ke[i + 6] = self.cfg["mjw_joint_stiffness"][i]
            b.joint_target_kd[i + 6] = self.cfg["mjw_joint_damping"][i]
            b.joint_armature[i + 6] = self.cfg["mjw_joint_armature"][i]
            b.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        b.joint_q[:3] = [0.0, 0.0, 0.0]
        b.add_ground_plane()
        if args.rigid:
            b.joint_q[2] = self._stand_height(b) + 0.005
        else:
            b.joint_q[2] = depth + self._stand_height(b) + 0.005
            # the hard floor under the bed holds the sand and, where the sand is thin, the robot too
            b.shape_flags[-1] = b.shape_flags[-1] | int(newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.COLLIDE_SHAPES)
            lo = np.array([-args.patch[0] / 2, -0.4, 0.0])
            hi = np.array([args.patch[0] / 2, -0.4 + args.patch[1], depth])
            wall = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=True)
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            hx, hy, t = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, 0.02
            for x, y, sx, sy in ((lo[0] - t, cy, t, hy + 2 * t), (hi[0] + t, cy, t, hy + 2 * t),
                                 (cx, lo[1] - t, hx, t), (cx, hi[1] + t, hx, t)):
                w = b.add_shape_box(body=-1, xform=wp.transform(wp.vec3(x, y, depth), wp.quat_identity()), hx=sx,
                                    hy=sy, hz=depth, cfg=wall)
                b.shape_flags[w] = b.shape_flags[w] & ~newton.ShapeFlags.VISIBLE   # keep the sand bed in view
            res = np.array(np.ceil(args.ppc * (hi - lo) / args.voxel), dtype=int)
            cell = (hi - lo) / res
            self.bed_lo, self.bed_hi, self.spacing = lo, hi, float(np.max(cell))
            n0 = len(b.particle_q)
            b.add_particle_grid(pos=wp.vec3(lo), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=res[0] + 1,
                                dim_y=res[1] + 1, dim_z=res[2] + 1, cell_x=cell[0], cell_y=cell[1], cell_z=cell[2],
                                mass=float(np.prod(cell) * args.density), jitter=0.0,
                                radius_mean=float(np.max(cell) * 0.5))
            # Newton seeds its own jitter from the particle count, so every run of a configuration would get the same
            # sand sample; jitter here with --seed so repeats sample different random packings.
            rng = np.random.default_rng(args.seed)
            q = np.asarray(b.particle_q[n0:], dtype=np.float64)
            q += (rng.random(q.shape) - 0.5) * float(np.max(cell))
            b.particle_q[n0:] = q.tolist()
        self.model = b.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.robot_mass = float(self.model.body_mass.numpy()[self.robot_bodies].sum())

        self.state = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        if args.rigid:
            self.solver = SolverMuJoCo(self.model, solver="newton", nconmax=30, njmax=100)
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
            self.mpm = None
        else:
            self.model.mpm.friction.fill_(args.friction)
            if args.cohesion > 0.0:   # damp sand; knob 4000 Pa ~ 1 kPa Terzaghi cohesion
                self.model.mpm.yield_stress.fill_(args.cohesion)
            if args.snow:   # compactable snow (calibrated against Bekker snow data)
                self.model.mpm.yield_pressure.fill_(args.yield_pressure)
                self.model.mpm.hardening.fill_(args.hardening)
                self.model.mpm.tensile_yield_ratio.fill_(args.tensile)
                self.model.mpm.dilatancy.fill_(1.0)
                # initial plastic volume ratio; must be set on the model (the coupled MPM entry builds its own state
                # from it). 1.0 = freshly deposited, consolidating under its own weight, as in the calibration.
                self.model.mpm.particle_Jp.fill_(args.jp0)
            mpm = SolverImplicitMPM.Config()
            mpm.voxel_size, mpm.tolerance, mpm.grid_type = args.voxel, args.tol, "sparse"
            mpm.strain_basis, mpm.max_iterations, mpm.critical_fraction = "P0", args.mpm_iters, 0.0
            mpm.velocity_basis, mpm.collider_basis = "Q1", "S2"
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, solver="newton",
                                                                 njmax=100),
                        bodies=self.robot_bodies, joints=self.robot_joints, substeps=args.robot_substeps),
                    SolverCoupledProxy.Entry(
                        name="mpm", solver=lambda v: SolverImplicitMPM(model=v, config=mpm),
                        particles=list(range(self.model.particle_count)), in_place=True),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[SolverCoupledProxy.Proxy(
                        source="mjc", destination="mpm", bodies=self.robot_bodies, mode="lagged",
                        collision_pipeline=lambda _model: None)],
                    iterations=args.proxy_iterations))
            self.mpm = self.solver.solver("mpm")
            self.collision = newton.CollisionPipeline(self.model, soft_contact_max=0)
            self.contacts = self.collision.contacts()
            slot = np.full(self.model.body_count, -1, dtype=np.int32)
            for k, f in enumerate(self.feet):
                slot[f] = k
            slot[0] = 4   # base: belly contact
            self.body_slot = wp.array(slot, dtype=wp.int32)
            self.foot_force = wp.zeros(5, dtype=wp.vec3)

        if args.policy:   # a fine-tuned TorchScript policy (48 -> 12), e.g. from train_sand.py
            import torch

            self.torch_policy = torch.jit.load(args.policy).cuda().eval()
            self.policy = None
        else:
            self.torch_policy = None
            self.policy = OnnxRuntime(str(asset / "rl_policies" / "mjw_go2.onnx"), device=self.device)
            self.p_in, self.p_out = self.policy.input_names[0], self.policy.output_names[0]
        self.q_init = wp.clone(self.state.joint_q[7:])
        self.ident = wp.array(np.arange(nd, dtype=np.int32), dtype=wp.int32)
        self.obs = wp.zeros((1, 12 + 3 * nd), dtype=wp.float32)
        self.prev_act = wp.zeros((1, nd), dtype=wp.float32)
        self.nd = nd
        self.log = {k: [] for k in ("t", "base_q", "base_qd", "feet", "feet_v", "foot_force", "joint_q", "joint_qd",
                                    "target", "command")}
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def _stand_height(self, b):
        """Base height at which the lowest foot sphere just touches z = 0 in the default pose."""
        m = b.finalize(requires_grad=False)
        s = m.state()
        newton.eval_fk(m, m.joint_q, m.joint_qd, s)
        q = s.body_q.numpy()
        return -min(q[f][2] for f in self.feet) + FOOT_RADIUS

    def command(self):
        """Forward speed after the settle time; yaw rate holds the heading along +y; sideways speed holds x = 0."""
        q = self.state.body_q.numpy()[0][3:]
        yaw = np.arctan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        err = (np.pi / 2 - yaw + np.pi) % (2 * np.pi) - np.pi
        walk = self.sim_time > self.args.settle
        # lateral hold: steer back to the bed's centreline x = 0 (body-frame sideways command), so a drifting robot
        # never reaches the edge of the sand bed
        vy = float(np.clip(self.args.lateral_gain * self.state.body_q.numpy()[0][0] * np.sin(yaw), -0.25, 0.25))
        return wp.vec3(self.args.speed if walk else 0.0, vy if walk else 0.0,
                       float(np.clip(2.0 * err, -0.5, 0.5)) if walk else 0.0)

    def step(self):
        wp.launch(_compute_obs_kernel, dim=1,
                  inputs=[self.state.joint_q, self.state.joint_qd, self.q_init, self.ident, wp.vec3(0.0, 0.0, -1.0),
                          self.command(), self.prev_act, self.nd, self.obs])
        if self.torch_policy is not None:
            import torch

            with torch.no_grad():
                act = wp.from_torch(self.torch_policy(wp.to_torch(self.obs)).contiguous())
        else:
            act = self.policy({self.p_in: self.obs})[self.p_out]
        wp.launch(_build_joint_target_q_kernel, dim=7 + self.nd,
                  inputs=[act, self.q_init, self.ident, float(self.cfg["action_scale"]), 7, self.control.joint_target_q])
        wp.copy(self.prev_act, act)
        if self.mpm is not None:
            self.foot_force.zero_()
        for _ in range(self.steps_per_policy):
            if self.args.rigid:
                for _ in range(self.args.robot_substeps):
                    self.state.clear_forces()
                    self.solver.step(self.state, self.state_1, self.control, None, self.dt / self.args.robot_substeps)
                    self.state, self.state_1 = self.state_1, self.state
            else:
                self.state.clear_forces()
                self.collision.collide(self.state, self.contacts)
                self.solver.step(self.state, self.state, self.control, self.contacts, self.dt)
                imp, _pos, cid = self.mpm.collect_collider_impulses(self.state)
                wp.launch(_body_force_kernel, dim=imp.shape[0],
                          inputs=[imp, cid, self.mpm.collider_body_index, self.body_slot,
                                  1.0 / (self.dt * self.steps_per_policy), self.foot_force])
        self.sim_time += self.policy_dt
        if self.args.log:
            self._log()

    def _log(self):
        q, qd = self.state.body_q.numpy(), self.state.body_qd.numpy()
        self.log["t"].append(self.sim_time)
        self.log["base_q"].append(q[0])
        self.log["base_qd"].append(qd[0])
        self.log["feet"].append(q[self.feet, :3])
        self.log["feet_v"].append(qd[self.feet, :3])
        self.log["foot_force"].append(self.foot_force.numpy() if self.mpm is not None else np.zeros((5, 3)))
        self.log["joint_q"].append(self.state.joint_q.numpy()[7:])
        self.log["joint_qd"].append(self.state.joint_qd.numpy()[6:])
        self.log["target"].append(self.control.joint_target_q.numpy()[7:])
        self.log["command"].append(np.array(self.command()))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in self.log.items()}, robot_mass=self.robot_mass,
                 surface=self.surface, foot_radius=FOOT_RADIUS, rigid=self.args.rigid,
                 ke=np.array(self.cfg["mjw_joint_stiffness"]), kd=np.array(self.cfg["mjw_joint_damping"]),
                 effort_limit=self.model.joint_effort_limit.numpy()[6:], speed=self.args.speed,
                 friction=self.args.friction, density=self.args.density, voxel=self.args.voxel,
                 sand_depth=self.args.sand_depth, cohesion=self.args.cohesion, seed=self.args.seed,
                 snow=self.args.snow, yield_pressure=self.args.yield_pressure, hardening=self.args.hardening,
                 jp0=self.args.jp0,
                 mpm_iters=self.args.mpm_iters)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--rigid", action="store_true", help="rigid ground with MuJoCo contacts (baseline)")
        p.add_argument("--seconds", type=float, default=6.0)
        p.add_argument("--settle", type=float, default=1.0, help="stand still this long before walking")
        p.add_argument("--speed", type=float, default=0.6)
        p.add_argument("--lateral-gain", type=float, default=2.0, help="[1/s]; 0 disables the centreline hold")
        p.add_argument("--dt", type=float, default=0.01, help="coupled sand/robot step")
        p.add_argument("--robot-substeps", type=int, default=2)
        p.add_argument("--voxel", type=float, default=0.03)
        p.add_argument("--ppc", type=float, default=2.0)
        p.add_argument("--sand-depth", type=float, default=0.25)
        p.add_argument("--patch", type=float, nargs=2, default=[1.0, 4.5])
        p.add_argument("--density", type=float, default=1600.0)
        p.add_argument("--friction", type=float, default=0.75)
        p.add_argument("--seed", type=int, default=0, help="random packing of the sand particles")
        p.add_argument("--snow", default="", choices=["", "settled", "powder", "packed"],
                       help="compactable snow instead of sand: settled = calibrated against Bekker snow data; "
                            "powder / packed = softer / firmer extrapolations")
        p.add_argument("--yield-pressure", type=float, default=5.0e4, help="snow compaction pressure scale [Pa]")
        p.add_argument("--hardening", type=float, default=10.0, help="snow compaction hardening")
        p.add_argument("--tensile", type=float, default=0.2, help="snow tensile / compressive yield ratio")
        p.add_argument("--jp0", type=float, default=1.0, help="snow initial plastic volume ratio")
        p.add_argument("--cohesion", type=float, default=0.0, help="mpm yield_stress [Pa]; 4000 ~ 1 kPa (damp sand)")
        p.add_argument("--mpm-iters", type=int, default=500)
        p.add_argument("--tol", type=float, default=1.0e-6)
        p.add_argument("--proxy-iterations", type=int, default=1)
        p.add_argument("--policy", default="", help="TorchScript policy instead of Newton's mjw_go2.onnx")
        p.add_argument("--log", default="")
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    ex = Example(viewer, args)
    n = int(round(args.seconds / ex.policy_dt))
    t1 = None
    for k in range(n):
        if k == 3:
            wp.synchronize()
            t1 = time.perf_counter()
        ex.step()
        ex.render()
        if (k + 1) % 50 == 0:
            q = ex.state.body_q.numpy()[0]
            print(f"t {ex.sim_time:4.1f} s  base ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:.3f})", flush=True)
    wp.synchronize()
    rate = (n - 3) * ex.policy_dt / (time.perf_counter() - t1)
    print(f"{ex.model.particle_count:,} sand particles, robot {ex.robot_mass:.2f} kg: {rate:.3f}x real time",
          flush=True)
    if args.log:
        ex.save(args.log)
