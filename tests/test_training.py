"""Training observations/rewards and opt-in CUDA reset/contact checks."""

import math
import os

import numpy as np
import pytest

from go2_rocky.training_config import TrainingConfig


@pytest.mark.parametrize("kwargs", ({"num_envs": 0}, {"max_slope_deg": 90},
                                    {"speed_min": -1}, {"episode_s": float("nan")},
                                    {"stage": "sand"}, {"friction": 0}))
def test_training_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_training_observations_match_inference_kernel():
    torch = pytest.importorskip("torch")
    import warp as wp
    from go2_rocky.rewards import actor_observation
    from go2_rocky.run_go2 import observe
    rng = np.random.default_rng(5)
    q = rng.normal(size=19).astype(np.float32)
    q[3:7] /= np.linalg.norm(q[3:7])
    qd = rng.normal(size=18).astype(np.float32)
    rest = rng.normal(size=12).astype(np.float32)
    previous = rng.normal(size=(1, 12)).astype(np.float32)
    command = (0.6, -0.2, 0.3)
    expected = wp.zeros((1, 48), device="cpu")
    wp.launch(observe, dim=1, device="cpu", inputs=[wp.array(q, device="cpu"), wp.array(qd, device="cpu"),
              wp.array(rest, device="cpu"), wp.array(previous, device="cpu"), wp.vec3(*command), expected])
    actual = actor_observation(torch.tensor(q)[None], torch.tensor(qd)[None], torch.tensor(rest),
                               torch.tensor(command)[None], torch.tensor(previous))
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)
    assert torch.linalg.vector_norm(actual[:, 6:9]) == pytest.approx(1)


def test_reward_allows_slope_alignment_and_rewards_forward_motion():
    torch = pytest.importorskip("torch")
    from go2_rocky.rewards import reward_terms
    angle = math.radians(25)
    q, qd = torch.zeros(2, 19), torch.zeros(2, 18)
    q[:, 3] = math.sin(angle/2)
    q[:, 6] = math.cos(angle/2)
    normal = torch.tensor([[0, -math.sin(angle), math.cos(angle)]]).expand(2, -1)
    tangent = torch.tensor([[0, math.cos(angle), math.sin(angle)]]).expand(2, -1)
    across = torch.tensor([[1., 0, 0]]).expand(2, -1)
    qd[1, :3] = tangent[1]*0.6
    zeros = torch.zeros(2, 12)
    terms = reward_terms(q, qd, zeros, zeros, zeros, torch.zeros(12),
                         torch.ones(12)*50, torch.ones(12), torch.ones(2, 12)*30,
                         tangent, across, normal, torch.ones(2)*0.6,
                         torch.zeros(2, 3), zeros, torch.ones(2)*0.34)
    torch.testing.assert_close(terms["posture"], torch.zeros(2), atol=1e-7, rtol=0)
    assert terms["normal_velocity"][1] == pytest.approx(0, abs=1e-7)
    assert sum(terms.values())[1] > sum(terms.values())[0]


def test_training_height_query_matches_native_triangle_convention():
    torch = pytest.importorskip("torch")
    from go2_rocky.rewards import grid_height
    heights = torch.tensor([[0., 1.], [2., 4.]])
    xy = torch.tensor([[0.25, -0.25], [-0.25, 0.25]])
    torch.testing.assert_close(grid_height(heights, xy, size=1), torch.tensor([1.5, 2.0]))


@pytest.mark.skipif(os.environ.get("RUN_TRAINING_GPU") != "1", reason="opt-in CUDA physics test")
def test_gpu_worlds_geometry_resets_timeouts_and_actor_initialization(tmp_path):
    import torch
    import warp as wp
    from argparse import Namespace
    from rsl_rl.runners import OnPolicyRunner
    from go2_rocky.rl_env import RockyEnv
    from go2_rocky.train_ppo import runner_config, initialize_actor, export_actor
    torch.set_num_threads(4)
    with wp.ScopedDevice("cuda:0"):
        env = RockyEnv(TrainingConfig(num_envs=4, stage="rocky", episode_s=1, observation_noise=False,
                                     surface_rockiness=1))
        obs = env.get_observations()
        assert obs["policy"].shape == (4, 48)
        assert obs["critic"].shape == (4, 52)
        np.testing.assert_allclose(env.model.gravity.numpy(), [[0, 0, -9.81]]*5, atol=1e-6)
        # Confirm real per-world static rock transforms reached the native solver.
        native_map = env.solver.mjc_geom_to_newton_shape.numpy()
        sizes = env.solver.mjw_model.geom_pos.shape
        assert sizes[0] == 4
        assert not np.allclose(env.solver.mjw_model.geom_pos.numpy()[1],
                               env.solver.mjw_model.geom_pos.numpy()[2])
        original = env.q.clone()
        env.q[0] = float("nan")
        env.reset_idx(torch.tensor([0], device="cuda:0"))
        assert torch.isfinite(env.q).all()
        torch.testing.assert_close(env.q[1:], original[1:])
        assert native_map.shape[0] == 4
        args = Namespace(rollout_steps=24, name="test", init_std=0.2, lr=1e-4)
        runner = OnPolicyRunner(env, runner_config(args), device="cuda:0")
        from pathlib import Path
        initialize_actor(runner, Path(env.asset_directory)/"rl_policies/mjw_go2.pt")
        env.episode_length_buf[:] = env.max_episode_length-1
        with torch.inference_mode():
            action = runner.alg.actor(env.get_observations())
            next_obs, reward, done, extras = env.step(action)
        assert done.all()
        assert extras["time_outs"].all()
        assert (env.episode_length_buf == 0).all()
        assert torch.isfinite(next_obs["policy"]).all() and torch.isfinite(reward).all()
        # Exercise contacts after settling, not only the airborne first step.
        with torch.inference_mode():
            for _ in range(40):
                next_obs, reward, done, _ = env.step(runner.alg.actor(next_obs))
        assert torch.isfinite(env.q).all() and torch.isfinite(reward).all()
        assert (env._clearance(env._path_coordinates()) > 0.18).all()
        export_actor(runner, tmp_path)
        assert (tmp_path/"policy.onnx").exists()
