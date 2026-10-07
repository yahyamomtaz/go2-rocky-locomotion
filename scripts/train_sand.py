"""Fine-tune the Go2 walking policy for sand with PPO (rsl-rl 5.5) on the fast sand model.

    uv run python scripts/train_sand.py --types f0.6 f0.75 f0.9 f1.0 f0.75_c2000 f0.75_c4000 f0.75_c8000 --name sand_v1

Starts from Newton's pretrained mjw_go2 actor (same network as Isaac Lab's Go2 flat task), warms the critic up with
the actor frozen, then trains both. Each episode draws a world: rigid ground (20%) or one of the 7 MPM-calibrated sand
types with bearing scale 0.1-0.4, drag scale 0.7-1.5 and hole keep 0.6-1.0 (deliberately covering softer sand
than the calibration centre 0.25 / 1.0 / 0.8, because the proxy is easier on the robot than the MPM sand), on a fixed
per-world sand thickness of 8-50 cm. The actor sees only what the real
robot can sense; the critic also gets the sand parameters. The result is exported as TorchScript (48 -> 12), the same
format as mjw_go2.pt, for evaluation on the MPM sand with go2_sand.py --policy.
"""
import argparse
import os
import time

import torch
import warp as wp

import newton.utils
from newton_terrain.go2_proxy_env import Go2SandEnv
from rsl_rl.runners import OnPolicyRunner


def runner_cfg(a):
    return {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "experiment_name": "go2_sand",
        "run_name": a.name,
        "logger": "tensorboard",
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["privileged"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [128, 128, 128], "activation": "elu",
                  "obs_normalization": False,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": a.init_std}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "elu",
                   "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                      "entropy_coef": 0.005, "num_learning_epochs": 5, "num_mini_batches": 4,
                      "learning_rate": a.lr, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01,
                      "max_grad_norm": 1.0},
        "multi_gpu": None,
    }


def load_pretrained_actor(runner, path=""):
    ts = torch.jit.load(path or str(newton.utils.download_asset("unitree_go2") / "rl_policies" / "mjw_go2.pt"))
    src = {k[len("actor."):]: v for k, v in ts.state_dict().items() if k.startswith("actor.")}
    mlp = runner.alg.actor.mlp
    mlp.load_state_dict(src)
    print("loaded pretrained actor weights:", list(src))


def set_actor_trainable(runner, flag):
    for p in runner.alg.actor.parameters():
        p.requires_grad_(flag)


def export(runner, path):
    actor = runner.alg.actor
    mlp = actor.mlp

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Sequential(*[m for m in mlp]).cpu()

        def forward(self, x):
            return self.actor(x)

    torch.jit.script(Policy()).save(path)
    print("exported", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=100, help="critic-only iterations with the actor frozen")
    ap.add_argument("--calib", type=float, nargs=3, default=[0.5, 1.5, 0.7], help="bearing, drag, hole keep")
    ap.add_argument("--sb-range", type=float, nargs=2, default=[0.1, 0.4], help="bearing scale range")
    ap.add_argument("--sh-range", type=float, nargs=2, default=[0.7, 1.5], help="drag scale range")
    ap.add_argument("--keep-range", type=float, nargs=2, default=[0.6, 1.0], help="hole memory range")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--rigid-frac", type=float, default=0.2)
    ap.add_argument("--name", default="sand_v1")
    ap.add_argument("--table", default="runs/proxy_table.json", help="proxy coefficients per terrain type")
    ap.add_argument("--types", nargs="*", default=[], help="train only on these terrain types (+ rigid)")
    ap.add_argument("--snow-sb-range", type=float, nargs=2, default=None, help="bearing scale range for snow")
    ap.add_argument("--snow-sh-range", type=float, nargs=2, default=[1.0, 1.0], help="drag scale range for snow")
    ap.add_argument("--term-height", type=float, default=0.10, help="end the episode below this body height [m]")
    ap.add_argument("--init-policy", default="", help="TorchScript actor to start from (default: Newton mjw_go2)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    wp.config.quiet = True
    torch.manual_seed(a.seed)
    env = Go2SandEnv(num_envs=a.envs, calib=tuple(a.calib), rigid_frac=a.rigid_frac, seed=a.seed,
                     train_ranges=(tuple(a.sb_range), tuple(a.sh_range), tuple(a.keep_range)), table_path=a.table,
                     term_height=a.term_height, type_filter=a.types or None,
                     snow_ranges=(tuple(a.snow_sb_range), tuple(a.snow_sh_range)) if a.snow_sb_range else None)
    log_dir = os.path.abspath(f"runs/train/{a.name}")
    runner = OnPolicyRunner(env, runner_cfg(a), log_dir=log_dir, device="cuda:0")
    load_pretrained_actor(runner, a.init_policy)
    t0 = time.time()
    set_actor_trainable(runner, False)
    runner.learn(a.warmup, init_at_random_ep_len=True)
    set_actor_trainable(runner, True)
    runner.learn(a.iters - a.warmup)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    export(runner, os.path.join(log_dir, "policy.pt"))
    print(f"trained {a.iters} iterations in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
