"""Fine-tune the Go2 actor on rigid slopes/rocks, or benchmark PPO throughput.

uv run --extra train python -m go2_rocky.train_ppo --benchmark --envs 256
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner

from go2_rocky.rl_env import RockyEnv
from go2_rocky.training_config import TrainingConfig


@contextmanager
def record_failure(directory):
    try:
        yield
    except BaseException as error:
        path = directory / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            path.write_text(json.dumps(manifest, indent=2)+"\n")
        raise


def runner_config(args):
    return {
        "num_steps_per_env": args.rollout_steps, "save_interval": 100,
        "experiment_name": "go2_rigid_climbing", "run_name": args.name,
        "logger": "tensorboard", "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [128, 128, 128],
                  "activation": "elu", "obs_normalization": False,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": args.init_std}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256, 128],
                   "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "learning_rate": args.lr, "schedule": "adaptive",
                      "desired_kl": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4,
                      "clip_param": 0.2, "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "entropy_coef": 0.005, "gamma": 0.99, "lam": 0.95, "max_grad_norm": 1.0},
        "multi_gpu": None,
    }


def initialize_actor(runner, path):
    source = torch.jit.load(str(path), map_location="cpu")
    weights = {key.removeprefix("actor."): value for key, value in source.state_dict().items()
               if key.startswith("actor.")}
    if not weights:
        raise ValueError("--init-policy must be a TorchScript actor with actor.* weights")
    runner.alg.actor.mlp.load_state_dict(weights, strict=True)
    # Check the entire observation ordering/MLP, not only tensor dimensions.
    x = torch.randn(16, 48)
    with torch.no_grad():
        expected = source(x)
        actual = runner.alg.actor.mlp(x.to(runner.device)).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    paired_onnx = Path(path).with_suffix(".onnx")
    if paired_onnx.exists():
        verify_onnx_actor(runner.alg.actor.mlp, paired_onnx, runner.device)


def verify_onnx_actor(actor, path, device):
    """Compare the actor with the actual evaluation runtime/checkpoint."""
    from warp_nn.runtime import OnnxRuntime
    runtime = OnnxRuntime(str(path), device=device)
    random = np.random.default_rng(701).normal(size=(1, 48)).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(random).to(next(actor.parameters()).device)).cpu().numpy()
    actual = runtime({runtime.input_names[0]: wp.array(random, device=device)})[runtime.output_names[0]].numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


def export_actor(runner, directory):
    """Export a fixed-batch, basic-op ONNX model supported by Warp-NN."""
    import copy
    actor = copy.deepcopy(runner.alg.actor.mlp).cpu().eval()
    class Actor(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.actor = network

        def forward(self, observation):
            return self.actor(observation)
    exported = Actor(actor)
    torch.jit.trace(exported, torch.zeros(1, 48)).save(str(directory / "policy.pt"))
    onnx_path = directory / "policy.onnx"
    torch.onnx.export(exported, torch.zeros(1, 48), str(onnx_path),
                      input_names=["observation"], output_names=["action"],
                      opset_version=17, dynamo=False)
    verify_onnx_actor(exported, onnx_path, runner.device)
    return hashlib.sha256(onnx_path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--stage", choices=("flat", "slope", "rocky"), default="slope")
    parser.add_argument("--max-slope", type=float, default=5)
    parser.add_argument("--roughness", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--rock-count", type=int, default=32)
    parser.add_argument("--surface-rockiness", type=float, default=0,
                        help="native fractured heightfield relief, in addition to convex rocks")
    parser.add_argument("--episode-seconds", type=float, default=20)
    parser.add_argument("--speed-range", type=float, nargs=2, default=(0.3, 0.8))
    parser.add_argument("--iters", type=int, default=1500, help="additional PPO iterations, including critic warmup")
    parser.add_argument("--warmup", type=int, default=25, help="critic-only iterations; 0 on a full resume")
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-std", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--name", default="slopes5_seed11")
    parser.add_argument("--output", type=Path, default=Path("runs/train"))
    parser.add_argument("--init-policy", type=Path, help="TorchScript actor for a new terrain stage")
    parser.add_argument("--resume", type=Path, help="full PPO model checkpoint (same training stage)")
    parser.add_argument("--benchmark", action="store_true", help="time PPO updates, save a test checkpoint/export")
    parser.add_argument("--benchmark-iters", type=int, default=100,
                        help="measured updates; use enough to include episode resets")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0 or args.rollout_steps < 4 or args.benchmark_iters < 1:
        parser.error("iterations/rollout must be positive; warmup must be nonnegative")
    if args.resume and args.init_policy:
        parser.error("choose --resume or --init-policy")
    if not 0 < args.lr <= 0.01 or not 0 < args.init_std <= 1:
        parser.error("invalid learning rate or action noise")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("this entrypoint supports one GPU per process; use independent jobs for training seeds")
    return args


def main():
    args = parse_args()
    config = TrainingConfig(num_envs=args.envs, seed=args.seed, stage=args.stage,
                            max_slope_deg=args.max_slope, roughness=args.roughness, rock_count=args.rock_count,
                            surface_rockiness=args.surface_rockiness,
                            episode_s=args.episode_seconds, speed_min=args.speed_range[0],
                            speed_max=args.speed_range[1], observation_noise=not args.no_noise,
                            capture=not args.no_capture)
    directory = args.output / args.name
    directory.mkdir(parents=True, exist_ok=True)
    # A run directory is never silently reused or overwritten.
    if (directory / "manifest.json").exists():
        raise FileExistsError(f"run exists: {directory}; choose a new --name (also when resuming)")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable; install the train extra and use an NVIDIA GPU node")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    wp.config.quiet = True
    started = time.perf_counter()
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    with record_failure(directory), wp.ScopedDevice(args.device):
        env = RockyEnv(config, device=args.device)
        cfg = runner_config(args)
        runner = OnPolicyRunner(env, cfg, log_dir=str(directory.resolve()), device=args.device)
        source = args.init_policy or Path(env.asset_directory) / "rl_policies/mjw_go2.pt"
        if args.resume:
            info = runner.load(str(args.resume), map_location=args.device)
            runner.current_learning_iteration = (info or {}).get("completed_iterations", runner.current_learning_iteration+1)
        else:
            initialize_actor(runner, source)
        manifest = {"training_config": asdict(config), "ppo": cfg,
                    "argv": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "gpu": torch.cuda.get_device_name(args.device),
                    "gpu_vram_bytes": torch.cuda.get_device_properties(args.device).total_memory,
                    "asset_directory": env.asset_directory,
                    "cuda_runtime": torch.version.cuda,
                    "gpu_compute_capability": torch.cuda.get_device_capability(args.device),
                    "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("train_ppo.py", "rl_env.py", "rewards.py", "training_config.py")},
                    "initial_weights": str(args.resume or source),
                    "initial_weights_sha256": hashlib.sha256((args.resume or source).read_bytes()).hexdigest(),
                    "terrain_bank": env.terrain_manifest, "rock_layouts": env.rock_layouts,
                    "gravity": [0, 0, -9.81], "physics_dt_s": env.physics_dt, "policy_dt_s": env.step_dt,
                    "training_goal_distance_m": 5,
                    "critic_warmup_lr_schedule": "fixed; adaptive KL scheduling starts when actor is unfrozen",
                    "height_query": "exact heightfield triangles; stability relative to mean slope normal",
                    "surface_sha256": hashlib.sha256(env.surface_heights.tobytes()).hexdigest(),
                    "versions": {p: version(p) for p in ("torch", "rsl-rl-lib", "newton", "warp-lang", "mujoco-warp")},
                    "status": "running", "benchmark_only": args.benchmark}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        np.savez_compressed(directory / "terrain.npz", surface_heights_m=env.surface_heights)
        setup_s = time.perf_counter()-started
        iterations = args.benchmark_iters+2 if args.benchmark else args.iters
        warmup = 0 if args.resume or args.benchmark else min(args.warmup, iterations)
        first_iteration = runner.current_learning_iteration
        log_iteration = runner.logger.log
        def log_full_budget(**values):
            # rsl-rl receives several short learn() calls for checkpointing.
            # Its ETA must use the entire run, not the end of each chunk.
            values.update(start_it=first_iteration, total_it=first_iteration+iterations)
            log_iteration(**values)
        runner.logger.log = log_full_budget
        completed = 0
        measured_s = 0.0
        measured_iters = 0
        used_gpu_memory = []
        actor_before = [p.detach().clone() for p in runner.alg.actor.mlp.parameters()]
        while completed < iterations and not stop["requested"]:
            frozen = completed < warmup
            # KL is identically zero for a frozen actor; adaptive scheduling
            # would otherwise raise LR throughout critic-only warmup.
            runner.alg.schedule = "fixed" if frozen else cfg["algorithm"]["schedule"]
            for parameter in runner.alg.actor.parameters():
                parameter.requires_grad_(not frozen)
            if args.benchmark and completed == 0:
                count = 2  # exclude first two PPO iterations from throughput timing
            elif frozen:
                count = min(25, warmup-completed)
            else:
                count = min(25, iterations-completed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                runner.learn(count)
            finally:
                # rsl-rl opens a new TensorBoard writer for each learn() call.
                if runner.logger.writer is not None:
                    runner.logger.writer.close()
            torch.cuda.synchronize()
            if frozen:
                for parameter, initial in zip(runner.alg.actor.mlp.parameters(), actor_before):
                    torch.testing.assert_close(parameter, initial, atol=0, rtol=0)
            seconds = time.perf_counter()-t0
            free_memory, total_memory = torch.cuda.mem_get_info(args.device)
            used_gpu_memory.append(total_memory-free_memory)
            if not args.benchmark or completed > 0:
                measured_s += seconds
                measured_iters += count
            completed += count
            # rsl-rl stores the last index, while the next chunk needs the next index.
            runner.current_learning_iteration += 1
            temporary_checkpoint = directory / "checkpoint.pending.pt"
            runner.save(str(temporary_checkpoint), infos={"completed_iterations": runner.current_learning_iteration})
            temporary_checkpoint.replace(directory / "checkpoint.pt")
            print(f"CHECKPOINT {directory / 'checkpoint.pt'} completed_this_run={completed}", flush=True)
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(True)
        onnx_sha = export_actor(runner, directory)
        transitions = measured_iters*args.envs*args.rollout_steps
        report = {"status": "interrupted" if stop["requested"] else "complete",
                  "benchmark_only": args.benchmark, "setup_seconds": setup_s,
                  "completed_iterations_this_run": completed, "measured_iterations": measured_iters,
                  "measured_seconds": measured_s, "seconds_per_iteration": measured_s/max(1, measured_iters),
                  "environment_transitions_per_second": transitions/max(1e-6, measured_s),
                  "estimated_1500_iterations_minutes": 1500*measured_s/max(1, measured_iters)/60,
                  "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_observed_total_gpu_used_bytes": max(used_gpu_memory, default=0),
                  "gpu_memory_note": "sampled between chunks; includes Warp, PyTorch and other GPU users",
                  "actor_max_parameter_change": max(float((p.detach()-b).abs().max()) for p, b in zip(runner.alg.actor.mlp.parameters(), actor_before)),
                  "completed_training_episodes": env.completed_episodes,
                  "training_successful_episodes": env.successful_episodes,
                  "training_fallen_episodes": env.fallen_episodes,
                  "policy_sha256": onnx_sha,
                  "note": "Throughput estimate, not time-to-convergence; training episodes are not held-out evaluation."}
        (directory / "timing.json").write_text(json.dumps(report, indent=2)+"\n")
        manifest.update(report)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
