"""Independent rigid-contact Go2 VecEnv using public Newton/rsl-rl APIs.

Training uses finite rigid slopes and optional convex rocks, vertical gravity,
and the evaluation runner's 48/12 interface. No sand model or MPM forces.
Static topology is shared; slope transforms and rock poses vary across worlds.
"""

from dataclasses import asdict
import math

import newton
import newton.utils
import numpy as np
import torch
from tensordict import TensorDict
import warp as wp
import yaml

from go2_rocky.drop_sphere import add_rigid_bed
from go2_rocky.rewards import actor_observation, reward_terms, rotate, rotate_inverse, grid_height
from go2_rocky.rocks import RockConfig, generate_rocks, add_static_rocks
from go2_rocky.terrain import SlopeTerrain
from go2_rocky.training_config import TrainingConfig
from go2_rocky.surface import RockSurface
from go2_rocky.mountain import MountainTerrain


class RockyEnv:
    num_actions = 12
    step_dt = 0.02
    physics_dt = 0.005
    decimation = 4

    def __init__(self, config=TrainingConfig(), device="cuda:0"):
        if not torch.cuda.is_available() or not wp.get_device(device).is_cuda:
            raise RuntimeError("Rigid PPO training requires a CUDA GPU; install uv sync --extra train")
        self.config, self.cfg = config, asdict(config)
        self.device, self.num_envs = device, config.num_envs
        self.rng = torch.Generator(device=device).manual_seed(config.seed)
        self.max_episode_length = int(config.episode_s / self.step_dt)
        asset = newton.utils.download_asset("unitree_go2")
        self.asset_directory = str(asset)
        policy_config = yaml.safe_load((asset / "rl_policies/go2.yaml").read_text())
        self.action_scale = float(policy_config["action_scale"])
        template = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=100, limit_kd=1)
        template.default_shape_cfg.ke = 5e4
        template.default_shape_cfg.kd = 500
        template.default_shape_cfg.kf = 1000
        template.default_shape_cfg.mu = config.friction
        template.rigid_gap = 0
        template.add_usd(str(asset / "usd/go2.usda"),
                         xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()),
                         collapse_fixed_joints=False, enable_self_collisions=False,
                         joint_ordering="dfs", hide_collision_shapes=True)
        template.approximate_meshes("convex_hull")
        ordering = [label.rsplit("/", 1)[-1] for label, kind in zip(template.joint_label, template.joint_type)
                    if kind == newton.JointType.REVOLUTE]
        if ordering != policy_config["mjw_joint_names"]:
            raise ValueError("joint order differs from the pretrained actor")
        template.joint_q[:7] = [0, 0, 0, 0, 0, 0, 1]
        template.joint_q[7:] = policy_config["mjw_joint_pos"]
        for j in range(12):
            for name, values in (("joint_target_ke", "mjw_joint_stiffness"),
                                 ("joint_target_kd", "mjw_joint_damping"),
                                 ("joint_armature", "mjw_joint_armature")):
                getattr(template, name)[6+j] = policy_config[values][j]
            template.joint_target_mode[6+j] = int(newton.JointTargetMode.POSITION)
        probe_model = template.finalize(device="cpu")
        probe = probe_model.state()
        newton.eval_fk(probe_model, probe.joint_q, probe.joint_qd, probe)
        poses = probe.body_q.numpy()
        bodies = probe_model.shape_body.numpy()
        types = probe_model.shape_type.numpy()
        shapes = probe_model.shape_transform.numpy()
        sizes = probe_model.shape_scale.numpy()
        feet = [i for i, b in enumerate(bodies) if b >= 0
                and probe_model.body_label[b].endswith("_foot") and types[i] == int(newton.GeoType.SPHERE)]
        if len(feet) != 4:
            raise ValueError("expected four foot spheres")
        self.spawn_height = max(0.02+sizes[i, 0]-wp.transform_point(
            wp.transform(*poses[bodies[i]]), wp.vec3(*shapes[i, :3]))[2] for i in feet)
        yaw = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi/2)
        template.joint_q[:7] = [0, -4.2, self.spawn_height, *yaw]
        self.surface_heights = np.zeros((61, 61), dtype=np.float32)
        axis = np.linspace(-6, 6, 61)
        x, y = np.meshgrid(axis, axis)
        if config.surface_rockiness:
            relief = RockSurface(config.seed, 12, config.roughness).relief(x, y, np.abs(x))
            pad_distance = np.hypot(x, y+4.2)
            pad = np.clip((pad_distance-0.8)/0.4, 0, 1)
            self.surface_heights[:] = relief*pad*config.surface_rockiness
            field = newton.Heightfield(data=self.surface_heights, nrow=61, ncol=61, hx=6, hy=6)
            template.add_shape_heightfield(heightfield=field, label="training_fractured_surface")
        else:
            add_rigid_bed(template, SlopeTerrain(), 4, 12)
        surface = MountainTerrain.__new__(MountainTerrain)
        surface.n, surface.spacing = 61, 0.2
        surface.width_m = surface.length_m = 12
        surface.heights = self.surface_heights
        rock_ids = []
        rocks = ()
        self.rock_layouts = []
        if config.stage == "rocky":
            rock_config = RockConfig(seed=config.seed, rock_count=config.rock_count,
                                     roughness=config.roughness, start_clearance_m=2.4)
            rocks = generate_rocks(rock_config)
            rock_ids = add_static_rocks(template, SlopeTerrain(), rocks)
        # Each world is a genuine rotated slope, with world gravity unchanged.
        random = np.random.default_rng(config.seed)
        grades = random.uniform(0, config.max_slope_deg, config.num_envs)
        grades[::5] = 0  # preserve flat walking
        if config.stage == "flat":
            grades[:] = 0
        directions = random.choice(("uphill", "downhill", "cross"), config.num_envs, p=(0.8, 0.1, 0.1))
        terrains = [SlopeTerrain(float(grade), str(direction)) for grade, direction in zip(grades, directions)]
        rotations = [wp.quat_from_matrix(wp.mat33(*np.asarray(t.rotation_matrix).ravel())) for t in terrains]
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(template, config.num_envs,
                        xforms=[wp.transform(wp.vec3(0, 0, 0), r) for r in rotations])
        # Same mesh topology across worlds, independent seeded layouts/poses.
        if rock_ids:
            for world, rotation in enumerate(rotations):
                placements = []
                for shape, rock in zip(rock_ids, rocks):
                    radius = rock.bounding_radius_m
                    for attempt in range(10000):
                        u = random.uniform(-1.9+radius, 1.9-radius)
                        v = random.uniform(-2.8+radius, 5.5-radius)
                        if all(math.hypot(u-x, v-y) >= radius+r+0.025 for x, y, r in placements):
                            break
                    else:
                        raise ValueError("training rock field is too dense; reduce --rock-count")
                    placements.append((u, v, radius))
                    scene.shape_transform[world*template.shape_count+shape] = wp.transform_multiply(
                        wp.transform(wp.vec3(0, 0, 0), rotation),
                        wp.transform(wp.vec3(u, v, float(surface.height_at(u, v))
                                             - (0.4*rock.height_m if config.surface_rockiness else 0)),
                                     wp.quat_from_axis_angle(wp.vec3(0, 0, 1), random.uniform(-math.pi, math.pi))))
                self.rock_layouts.append(placements)
        self.terrain_manifest = [asdict(t) for t in terrains]
        self.model = scene.finalize(device=device)
        self.model.set_gravity((0, 0, -9.81))
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=True,
                                                solver="newton", nconmax=64, njmax=192)
        self.state_a, self.state_b = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.q = wp.to_torch(self.state_a.joint_q).view(config.num_envs, 19)
        self.qd = wp.to_torch(self.state_a.joint_qd).view(config.num_envs, 18)
        self.target = wp.to_torch(self.control.joint_target_q).view(config.num_envs, 19)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.frames = tensor([t.rotation_matrix for t in terrains])
        self.height_grid = tensor(self.surface_heights)
        self.across, self.tangent, self.normal = (self.frames[:, :, i] for i in range(3))
        self.bed_quat = tensor([list(r) for r in rotations])
        self.rest = tensor(policy_config["mjw_joint_pos"])
        self.ke = tensor(policy_config["mjw_joint_stiffness"])
        self.kd = tensor(policy_config["mjw_joint_damping"])
        self.effort = wp.to_torch(self.model.joint_effort_limit).view(config.num_envs, 18)[:, 6:]
        self.actions = torch.zeros((config.num_envs, 12), device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_qd = torch.zeros_like(self.actions)
        self.command = torch.zeros((config.num_envs, 3), device=device)
        self.speed = torch.zeros(config.num_envs, device=device)
        self.episode_length_buf = torch.zeros(config.num_envs, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(config.num_envs, device=device)
        self.start_progress = torch.zeros(config.num_envs, device=device)
        self.reset_mask = torch.zeros(config.num_envs+1, dtype=torch.bool, device=device)
        self.noise = tensor([0.05]*3+[0.1]*3+[0.025]*3+[0]*3+[0.005]*12+[0.5]*12+[0]*12)
        self.completed_episodes = self.successful_episodes = self.fallen_episodes = 0
        self.graph = None
        self.reset_idx(torch.arange(config.num_envs, device=device))
        if config.capture:
            with wp.ScopedCapture(device=device) as capture:
                self._physics()
            self.graph = capture.graph
            self.reset_idx(torch.arange(config.num_envs, device=device))

    def _physics(self):
        for _ in range(self.decimation):
            self.state_a.clear_forces()
            self.solver.step(self.state_a, self.state_b, self.control, None, self.physics_dt)
            self.state_a, self.state_b = self.state_b, self.state_a

    def _path_coordinates(self):
        return torch.einsum("nji,nj->ni", self.frames, self.q[:, :3])

    def _clearance(self, local):
        if not self.config.surface_rockiness:
            return local[:, 2]
        return local[:, 2]-grid_height(self.height_grid, local[:, :2])

    def _commands(self):
        axes = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3)
        forward = rotate(self.q[:, 3:7], axes[:, 0])
        left = rotate(self.q[:, 3:7], axes[:, 1])
        local = self._path_coordinates()
        lateral = (-0.8*local[:, 0]).clamp(-0.25, 0.25)
        desired = self.speed[:, None]*self.tangent + lateral[:, None]*self.across
        error = torch.atan2((forward*self.across).sum(-1), (forward*self.tangent).sum(-1))
        self.command[:, 0] = (desired*forward).sum(-1)
        self.command[:, 1] = (desired*left).sum(-1).clamp(-0.25, 0.25)
        self.command[:, 2] = (2*error-0.2*(self.qd[:, 3:6]*self.normal).sum(-1)).clamp(-0.8, 0.8)

    def reset_idx(self, ids, update_commands=True):
        if not ids.numel():
            return
        n = len(ids)
        random = lambda *shape: torch.rand(shape, generator=self.rng, device=self.device)
        local = torch.stack((0.5*(random(n)-0.5), -4.2+0.3*(random(n)-0.5),
                             torch.full((n,), self.spawn_height, device=self.device)), -1)
        self.q[ids, :3] = torch.bmm(self.frames[ids], local[:, :, None]).squeeze(-1)
        heading = math.pi/2+0.15*(random(n)-0.5)
        half = heading/2
        yaw = torch.stack((half*0, half*0, half.sin(), half.cos()), -1)
        # Quaternion product of terrain rotation and local yaw.
        a, b = self.bed_quat[ids], yaw
        xyz = a[:, 3:]*b[:, :3]+b[:, 3:]*a[:, :3]+torch.cross(a[:, :3], b[:, :3], dim=-1)
        scalar = a[:, 3:]*b[:, 3:]-(a[:, :3]*b[:, :3]).sum(-1, keepdim=True)
        self.q[ids, 3:7] = torch.cat((xyz, scalar), -1)
        self.q[ids, 7:] = self.rest  # valid settled stance, no initial foot penetration
        self.qd[ids] = 0
        self.target[ids, 7:] = self.rest
        self.actions[ids] = self.previous_actions[ids] = self.previous_qd[ids] = 0
        self.speed[ids] = self.config.speed_min+(self.config.speed_max-self.config.speed_min)*random(n)
        self.episode_length_buf[ids] = 0
        self.episode_return[ids] = 0
        self.start_progress[ids] = local[:, 1]
        self.reset_mask.zero_()
        self.reset_mask[ids] = True
        self.solver.reset(self.state_a, wp.from_torch(self.reset_mask, dtype=wp.bool), flags=0)
        # MuJoCo derives body poses from q on its next step. No custom forces
        # in this environment read stale body poses between reset and step.
        if update_commands:
            self._commands()

    def get_observations(self):
        obs = actor_observation(self.q, self.qd, self.rest, self.command, self.actions)
        local = self._path_coordinates()
        clearance = self._clearance(local)[:, None]
        privileged = torch.cat((obs, rotate_inverse(self.q[:, 3:7], self.normal),
                                clearance), -1)
        if self.config.observation_noise:
            obs = obs+(2*torch.rand(obs.shape, generator=self.rng, device=self.device)-1)*self.noise
        return TensorDict({"policy": obs, "critic": privileged}, batch_size=[self.num_envs])

    def step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.previous_qd.copy_(self.qd[:, 6:])
        self.actions.copy_(actions.clamp(-20, 20))
        self.target[:, 7:] = self.rest+self.action_scale*self.actions
        if self.graph is None:
            self._physics()
        else:
            wp.capture_launch(self.graph)
        self.episode_length_buf += 1
        local = self._path_coordinates()
        clearance = self._clearance(local)
        finite = torch.isfinite(self.q).all(-1) & torch.isfinite(self.qd).all(-1)
        world_up = torch.zeros_like(self.normal)
        world_up[:, 2] = 1
        alignment = (rotate(self.q[:, 3:7], world_up)*self.normal).sum(-1)
        fell = (clearance < 0.18) | (alignment < 0.4) | ~finite
        escaped = (local[:, 0].abs() > 1.4) | (local[:, 1].abs() > 5.4)
        success = (local[:, 1]-self.start_progress >= 5) & ~fell & ~escaped
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fell | escaped | success | timeout
        terms = reward_terms(self.q, self.qd, self.actions, self.previous_actions,
                             self.target[:, 7:], self.rest, self.ke, self.kd, self.effort,
                             self.tangent, self.across, self.normal, self.speed, self.command,
                             self.previous_qd, clearance)
        reward = torch.nan_to_num(sum(terms.values()), nan=-2, posinf=-2, neginf=-2)
        reward -= 2*fell.float()+0.5*escaped.float()
        reward += success.float()
        self.episode_return += reward
        ids = done.nonzero().flatten()
        extras = {"time_outs": timeout & ~(fell | escaped | success), "log": {}}
        if len(ids):
            self.completed_episodes += len(ids)
            self.successful_episodes += int(success[ids].sum())
            self.fallen_episodes += int(fell[ids].sum())
            extras["log"] = {"Episode/return": self.episode_return[ids].mean(),
                              "Episode/progress_m": (local[ids, 1]-self.start_progress[ids]).mean(),
                              "Episode/fall_fraction": fell[ids].float().mean(),
                              "Episode/success_fraction": success[ids].float().mean(),
                              **{f"Reward/{key}": value[ids].mean() for key, value in terms.items()}}
            self.reset_idx(ids, update_commands=False)
        self._commands()
        return self.get_observations(), reward, done, extras
