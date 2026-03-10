from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_run_dir(base="experiments", env_name="env", algo="algo", seed=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"_seed{seed}" if seed is not None else ""
    run_name = f"{env_name}_{algo}{seed_part}_{ts}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


algo_map = {"dqn": DQN, "ddpg": DDPG, "a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}


def make_sb3_single_agent(env, algo, device, verbose, load, n_envs, seed, tb_log, **kwargs):
    run_dir = make_run_dir(env_name=env, algo=algo)
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    mntr_dir = run_dir / "monitor"
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = dict(kwargs)
    if tb_log:
        model_kwargs["tensorboard_log"] = tb_dir

    if n_envs > 1:
        env = make_vec_env(
            env, n_envs=n_envs, seed=seed, monitor_dir=mntr_dir, vec_env_cls=SubprocVecEnv
        )

    algo = algo_map[algo.lower()]
    if load:
        agent = algo.load(load, env=env, device=device, **kwargs)
    else:
        agent = algo("CnnPolicy", env=env, device=device, verbose=verbose, **kwargs)

    return agent
