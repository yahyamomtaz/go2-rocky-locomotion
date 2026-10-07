"""Many Unitree Go2 worlds on the fast sand model (sand_proxy.py), MuJoCo Warp physics, as an rsl-rl VecEnv.

Mirrors Isaac Lab's Go2 flat velocity task (which the pretrained mjw_go2 policy comes from): 48-dim observation
(base lin/ang velocity, projected gravity, command, joint pos/vel, last action) with the same noise, joint position
actions (scale 0.25, PD 50/1), 5 ms physics, 50 Hz policy, same rewards. Added: per world a rigid floor at z = 0 and a
sand layer of thickness D on top (or no sand), sand type and force scales drawn per episode.
The critic also sees privileged terms (sand type one-hot, scales, depth, foot depths, base height).
"""
import json
import math

import numpy as np
import torch
import warp as wp
import yaml
from tensordict import TensorDict

import newton
import newton.utils
from newton_terrain import sand_proxy as SP
from newton import JointTargetMode
from newton.solvers import SolverMuJoCo

FEET = ("FL", "FR", "RL", "RR")


def quat_rotate_inv(q, v):
    """Rotate world vectors v into the frame of quaternions q (x, y, z, w), batched."""
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v - w * t + torch.cross(u, t, dim=-1)


def yaw_of(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Go2SandEnv:
    def __init__(self, num_envs=2048, device="cuda:0", rigid_frac=0.2, depth_range=(0.08, 0.5), seed=0,
                 episode_s=20.0, calib=(1.0, 1.0, 0.7), scale_jitter=0.2, eval_condition=None, train_ranges=None,
                 table_path="runs/proxy_table.json", snow_keep=(0.9, 1.0), term_height=0.10, snow_ranges=None,
                 type_filter=None):
        """eval_condition: dict(type=name or None for rigid, depth=, command=(vx, vy), seconds=) makes every world the
        same condition with a fixed command and no noise (for validating the model against MPM)."""
        self.num_envs, self.device = num_envs, device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.dt, self.decimation = 0.005, 4
        self.step_dt = self.dt * self.decimation
        self.max_episode_length = int(episode_s / self.step_dt)
        self.eval = eval_condition
        self.cfg = dict(num_envs=num_envs, rigid_frac=rigid_frac, depth_range=depth_range, calib=calib,
                        scale_jitter=scale_jitter, eval=eval_condition)
        asset = newton.utils.download_asset("unitree_go2")
        with open(asset / "rl_policies" / "go2.yaml", encoding="utf-8") as f:
            ycfg = yaml.safe_load(f)
        self.nd = nd = ycfg["num_dofs"]
        self.num_actions = nd
        self.action_scale = float(ycfg["action_scale"])

        go2 = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(go2)
        go2.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
        go2.default_shape_cfg.ke, go2.default_shape_cfg.kd, go2.default_shape_cfg.kf = 5.0e4, 5.0e2, 1.0e3
        go2.default_shape_cfg.mu = 0.75
        go2.rigid_gap = 0.0
        go2.add_usd(str(asset / "usd" / "go2.usda"), xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
                    collapse_fixed_joints=False, enable_self_collisions=False, joint_ordering="dfs",
                    hide_collision_shapes=True)
        go2.approximate_meshes("convex_hull")
        go2.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
        go2.joint_q[7:] = ycfg["mjw_joint_pos"]
        for i in range(nd):
            go2.joint_target_ke[i + 6] = ycfg["mjw_joint_stiffness"][i]
            go2.joint_target_kd[i + 6] = ycfg["mjw_joint_damping"][i]
            go2.joint_armature[i + 6] = ycfg["mjw_joint_armature"][i]
            go2.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        self.nb = go2.body_count
        lbl = [go2.body_label[i].split("/")[-1] for i in range(self.nb)]
        self.calf_local = [lbl.index(f"{f}_calf") for f in FEET]
        self.foot_local = [lbl.index(f"{f}_foot") for f in FEET]
        self.thigh_local = [lbl.index(f"{f}_thigh") for f in FEET]
        go2.joint_q[:3] = [0.0, 0.0, 0.0]
        tmp = go2.finalize(requires_grad=False)
        st = tmp.state()
        newton.eval_fk(tmp, tmp.joint_q, tmp.joint_qd, st)
        self.stand_h = float(-min(st.body_q.numpy()[f][2] for f in self.foot_local) + SP.R_FOOT + 0.01)

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(go2, num_envs)
        scene.add_ground_plane()
        self.model = scene.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        # startup randomization: base mass x [0.8, 1.25] (log-uniform), as in Isaac Lab
        bm = self.model.body_mass.numpy()
        bi = self.model.body_inertia.numpy()
        rs = np.random.default_rng(seed)
        k = np.exp(rs.uniform(np.log(0.8), np.log(1.25), num_envs)) if eval_condition is None else np.ones(num_envs)
        base_ids = np.arange(num_envs) * self.nb
        bm[base_ids] *= k
        bi[base_ids] *= k[:, None, None]
        self.model.body_mass.assign(bm)
        self.model.body_inertia.assign(bi)
        self.solver = SolverMuJoCo(self.model, solver="newton", nconmax=35, njmax=65)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # torch views
        T = wp.to_torch
        self.q = T(self.state_0.joint_q).view(num_envs, 7 + nd)
        self.qd = T(self.state_0.joint_qd).view(num_envs, 6 + nd)
        self.body_q = T(self.state_0.body_q).view(num_envs, self.nb, 7)
        self.target = T(self.control.joint_target_q).view(num_envs, 7 + nd)
        self.default_q = torch.tensor(ycfg["mjw_joint_pos"], device=device, dtype=torch.float32)
        self.ke = torch.tensor(ycfg["mjw_joint_stiffness"], device=device)
        self.kd = torch.tensor(ycfg["mjw_joint_damping"], device=device)
        self.effort = T(self.model.joint_effort_limit).view(num_envs, 6 + nd)[:, 6:].clone()

        # sand state (per world / per leg)
        n4 = num_envs * 4
        self.mem = wp.full(n4, -1.0, dtype=float)
        self.nh = 16
        self.hole_x = wp.zeros(num_envs * self.nh, dtype=float)
        self.hole_y = wp.zeros(num_envs * self.nh, dtype=float)
        self.hole_d = wp.zeros(num_envs * self.nh, dtype=float)
        self.hole_n = wp.zeros(num_envs, dtype=int)
        self.keep_wp = wp.zeros(num_envs, dtype=float)
        self.keep = T(self.keep_wp)
        self.contact_wp = wp.zeros(n4, dtype=float)
        self.depth_wp = wp.zeros(n4, dtype=float)
        self.contact = T(self.contact_wp).view(num_envs, 4)
        self.foot_depth = T(self.depth_wp).view(num_envs, 4)
        self.surf_wp = wp.zeros(num_envs, dtype=float)
        self.type_wp = wp.zeros(num_envs, dtype=int)
        self.sb_wp = wp.ones(num_envs, dtype=float)
        self.sh_wp = wp.ones(num_envs, dtype=float)
        self.surf, self.stype = T(self.surf_wp), T(self.type_wp)
        self.sb, self.sh = T(self.sb_wp), T(self.sh_wp)
        table = json.load(open(table_path))
        self.types = list(table)          # sand types, then snow types (names start with "snow_")
        self.snow_type = torch.tensor([t.startswith("snow") for t in self.types], device=device)
        self.snow_keep = snow_keep        # snow holds its footprints (cohesive), so hole memory stays near 1
        self.snow_ranges = snow_ranges    # (bearing, drag) scale ranges for snow types only (None: as sand)
        allowed = [i for i, t in enumerate(self.types) if type_filter is None or t in type_filter]
        self.allowed_types = torch.tensor(allowed, device=device, dtype=torch.int32)
        self.table = SP.make_table([table[t] for t in self.types])
        self.calib = calib
        self.scale_jitter = scale_jitter
        # training: draw bearing / drag scale and hole keep per episode from these ranges (overrides calib +- jitter)
        self.train_ranges = train_ranges
        # episode ends when the body is lower than this above the surface (0 lets it plough through deep snow)
        self.term_height = term_height
        self.depth_range = depth_range
        # fixed per world: rigid or sand, and sand thickness
        u = torch.rand(num_envs, generator=self.rng, device=device)
        self.is_rigid = u < rigid_frac
        self.thickness = depth_range[0] + (depth_range[1] - depth_range[0]) * torch.rand(
            num_envs, generator=self.rng, device=device)
        if eval_condition is not None:
            self.is_rigid[:] = eval_condition.get("type") is None
            self.thickness[:] = eval_condition.get("depth", 0.3)
        self.thickness[self.is_rigid] = 0.0
        self.surf.copy_(self.thickness)
        self._ids = dict(calf=wp.array(self.calf_local, dtype=int), foot=wp.array(self.foot_local, dtype=int),
                         thigh=wp.array(self.thigh_local, dtype=int))

        # buffers
        self.actions = torch.zeros(num_envs, nd, device=device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_qd = torch.zeros(num_envs, nd, device=device)
        self.cmd = torch.zeros(num_envs, 3, device=device)
        self.heading = torch.zeros(num_envs, device=device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cmd_timer = torch.zeros(num_envs, device=device)
        self.push_timer = torch.zeros(num_envs, device=device)
        self.air_time = torch.zeros(num_envs, 4, device=device)
        self.last_contact = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.ep_sums = {}
        self._graph = None
        self.reset_idx(torch.arange(num_envs, device=device))
        self._capture()

    # ------------------------------------------------------------------ physics
    def _physics(self):
        for _ in range(self.decimation):   # even count: state_0 holds the result
            self.state_0.clear_forces()
            wp.launch(SP.sand_forces, dim=self.num_envs * 4,
                      inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com, self.nb,
                              self._ids["calf"], self._ids["foot"], self._ids["thigh"], 0, self.surf_wp, self.type_wp,
                              self.sb_wp, self.sh_wp, self.table, self.keep_wp, self.nh, self.hole_x, self.hole_y,
                              self.hole_d, self.hole_n, self.mem, self.contact_wp, self.depth_wp,
                              self.state_0.body_f])
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._physics()
            self._graph = cap.graph

    # ------------------------------------------------------------------ resets
    def _resample_commands(self, ids):
        n = len(ids)
        if self.eval is not None:
            c = self.eval.get("command", (0.6, 0.0))
            self.cmd[ids, 0], self.cmd[ids, 1] = c[0], c[1]
            self.heading[ids] = 0.0
        else:
            r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=self.rng, device=self.device)
            self.cmd[ids, 0], self.cmd[ids, 1] = r(-1.0, 1.0), r(-1.0, 1.0)
            self.heading[ids] = r(-math.pi, math.pi)
            standing = torch.rand(n, generator=self.rng, device=self.device) < 0.02
            self.cmd[ids[standing]] = 0.0
            self.heading[ids[standing]] = yaw_of(self.q[ids[standing], 3:7])
        self.cmd_timer[ids] = 10.0

    def reset_idx(self, ids):
        if len(ids) == 0:
            return
        n, dev = len(ids), self.device
        r = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(n, *s, generator=self.rng, device=dev)
        ev = self.eval is not None
        # sand of this episode
        if ev:
            t = self.eval.get("type")
            self.stype[ids] = -1 if t is None else self.types.index(t)
            j = self.eval.get("jitter", 0.0)
            self.sb[ids] = self.calib[0] * r(1 - j, 1 + j)
            self.sh[ids] = self.calib[1] * r(1 - j, 1 + j)
            self.keep[ids] = self.calib[2]
        else:
            ty = self.allowed_types[torch.randint(0, len(self.allowed_types), (n,), generator=self.rng, device=dev)]
            ty[self.is_rigid[ids]] = -1
            self.stype[ids] = ty
            if self.train_ranges is not None:
                (b0, b1), (h0, h1), (k0, k1) = self.train_ranges
                self.sb[ids], self.sh[ids], self.keep[ids] = r(b0, b1), r(h0, h1), r(k0, k1)
                snow = self.snow_type[ty.clamp(min=0).long()] & (ty >= 0)
                self.keep[ids] = torch.where(snow, r(*self.snow_keep), self.keep[ids])
                if self.snow_ranges is not None:
                    (sb0, sb1), (sh0, sh1) = self.snow_ranges
                    self.sb[ids] = torch.where(snow, r(sb0, sb1), self.sb[ids])
                    self.sh[ids] = torch.where(snow, r(sh0, sh1), self.sh[ids])
            else:
                self.sb[ids] = self.calib[0] * r(1 - self.scale_jitter, 1 + self.scale_jitter)
                self.sh[ids] = self.calib[1] * r(1 - self.scale_jitter, 1 + self.scale_jitter)
                self.keep[ids] = torch.clamp(self.calib[2] * r(1 - self.scale_jitter, 1 + self.scale_jitter), 0.0, 1.0)
        mem = wp.to_torch(self.mem).view(self.num_envs, 4)
        mem[ids] = -1.0
        wp.to_torch(self.hole_d).view(self.num_envs, self.nh)[ids] = 0.0
        wp.to_torch(self.hole_x).view(self.num_envs, self.nh)[ids] = 1.0e3
        # robot
        q = self.q[ids]
        yaw = torch.zeros(n, device=dev) if ev else r(-math.pi, math.pi)
        q[:, 0] = 0.0
        q[:, 1] = 0.0
        q[:, 2] = self.surf[ids] + self.stand_h
        q[:, 3:7] = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], 1)
        q[:, 7:] = self.default_q * (1.0 if ev else r(0.5, 1.5, self.nd))
        self.q[ids] = q
        self.qd[ids] = 0.0 if ev else torch.cat([r(-0.5, 0.5, 6), torch.zeros(n, self.nd, device=dev)], 1)
        self.target[ids, 7:] = self.q[ids, 7:]
        self.actions[ids] = 0.0
        self.prev_actions[ids] = 0.0
        self.prev_qd[ids] = 0.0
        self.air_time[ids] = 0.0
        self.last_contact[ids] = False
        self.episode_length_buf[ids] = 0
        self.push_timer[ids] = r(10.0, 15.0)
        self._resample_commands(ids)
        # body poses must follow the new joint state before the sand model reads them on the next physics step
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    # ------------------------------------------------------------------ observations
    def _base(self):
        quat = self.q[:, 3:7]
        v_b = quat_rotate_inv(quat, self.qd[:, 0:3])
        w_b = quat_rotate_inv(quat, self.qd[:, 3:6])
        g_b = quat_rotate_inv(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3))
        return quat, v_b, w_b, g_b

    def get_observations(self):
        quat, v_b, w_b, g_b = self._base()
        jp = self.q[:, 7:] - self.default_q
        jv = self.qd[:, 6:]
        obs = torch.cat([v_b, w_b, g_b, self.cmd, jp, jv, self.actions], 1)
        if self.eval is None:   # Isaac Lab observation noise
            noise = torch.cat([torch.full((3,), 0.1), torch.full((3,), 0.2), torch.full((3,), 0.05), torch.zeros(3),
                               torch.full((self.nd,), 0.01), torch.full((self.nd,), 1.5),
                               torch.zeros(self.nd)]).to(self.device)
            obs_n = obs + (2 * torch.rand(obs.shape, generator=self.rng, device=self.device) - 1) * noise
        else:
            obs_n = obs
        onehot = torch.zeros(self.num_envs, len(self.types) + 1, device=self.device)
        onehot[torch.arange(self.num_envs), (self.stype.long() + 1)] = 1.0
        h = (self.q[:, 2] - self.surf).unsqueeze(1)
        priv = torch.cat([obs, onehot, self.sb.unsqueeze(1), self.sh.unsqueeze(1), self.surf.unsqueeze(1),
                          self.foot_depth, h, self.contact], 1)
        return TensorDict({"policy": obs_n, "privileged": priv}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------ step
    def step(self, actions):
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-100.0, 100.0))
        self.target[:, 7:] = self.default_q + self.action_scale * self.actions
        self.prev_qd.copy_(self.qd[:, 6:])
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._physics()
        self.episode_length_buf += 1
        quat, v_b, w_b, g_b = self._base()
        # heading command -> yaw rate
        yaw = yaw_of(quat)
        err = torch.atan2(torch.sin(self.heading - yaw), torch.cos(self.heading - yaw))
        self.cmd[:, 2] = torch.clamp(0.5 * err, -1.0, 1.0)
        rew, terms = self._rewards(v_b, w_b, g_b)
        # terminations: belly on the ground/sand or flipped
        h = self.q[:, 2] - self.surf
        died = (h < self.term_height) | (g_b[:, 2] > -0.5) | ~torch.isfinite(self.q).all(1)
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = died | timeout
        for k, v in terms.items():
            self.ep_sums[k] = self.ep_sums.get(k, torch.zeros(self.num_envs, device=self.device)) + v
        extras = {"time_outs": timeout & ~died, "log": {}}
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            for k, v in self.ep_sums.items():
                extras["log"][f"Episode_Reward/{k}"] = (v[done_ids].mean() / (self.max_episode_length * self.step_dt))
                v[done_ids] = 0.0
            extras["log"]["Episode_Termination/died"] = died.float().mean()
            if self.eval is None:
                self.reset_idx(done_ids)
        # command resampling and pushes
        self.cmd_timer -= self.step_dt
        ids = (self.cmd_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(ids) and self.eval is None:
            self._resample_commands(ids)
        self.push_timer -= self.step_dt
        ids = (self.push_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(ids) and self.eval is None:
            self.qd[ids, 0:2] += -0.5 + torch.rand(len(ids), 2, generator=self.rng, device=self.device)
            self.push_timer[ids] = 10.0 + 5.0 * torch.rand(len(ids), generator=self.rng, device=self.device)
        return self.get_observations(), rew, dones, extras

    def _rewards(self, v_b, w_b, g_b):
        dt = self.step_dt
        lin_err = ((self.cmd[:, :2] - v_b[:, :2]) ** 2).sum(1)
        ang_err = (self.cmd[:, 2] - w_b[:, 2]) ** 2
        q, qd = self.q[:, 7:], self.qd[:, 6:]
        tau = torch.clamp(self.ke * (self.target[:, 7:] - q) - self.kd * qd, -self.effort, self.effort)
        contact = self.contact > 0.5
        first = contact & ~self.last_contact
        self.air_time += dt
        air_rew = ((self.air_time - 0.5) * first).sum(1) * (self.cmd[:, :2].norm(dim=1) > 0.1)
        self.air_time[contact] = 0.0
        self.last_contact = contact
        terms = {
            "track_lin_vel_xy": 1.5 * torch.exp(-lin_err / 0.25),
            "track_ang_vel_z": 0.75 * torch.exp(-ang_err / 0.25),
            "lin_vel_z": -2.0 * v_b[:, 2] ** 2,
            "ang_vel_xy": -0.05 * (w_b[:, :2] ** 2).sum(1),
            "dof_torques": -2.0e-4 * (tau ** 2).sum(1),
            "dof_acc": -2.5e-7 * (((qd - self.prev_qd) / dt) ** 2).sum(1),
            "action_rate": -0.01 * ((self.actions - self.prev_actions) ** 2).sum(1),
            "feet_air_time": 0.25 * air_rew,
            "flat_orientation": -2.5 * (g_b[:, :2] ** 2).sum(1),
        }
        terms = {k: v * dt for k, v in terms.items()}
        return sum(terms.values()), terms
