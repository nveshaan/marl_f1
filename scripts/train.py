import argparse
import os
import sys

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baselines import make_sb3_single_agent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser(cfg) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start training an agent.")
    parser.add_argument("env", type=str, help="CarRacing-v3 or MultiCarRacing-v0")
    parser.add_argument("--algo", type=str, default="sac", choices=cfg["algos"])
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tb-log", action="store_true", help="Enable TensorBoard logging")
    return parser


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    cfg = load_config("configs/single_agent.yml")
    parser = build_parser(cfg)
    args = parser.parse_args()

    agent = make_sb3_single_agent(
        env=args.env,
        algo=args.algo,
        device=device,
        verbose=args.verbose,
        load=args.load,
        n_envs=args.n_envs,
        seed=args.seed,
        tb_log=args.tb_log,
    )

    agent.learn(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
