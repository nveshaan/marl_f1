import importlib
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)

importlib.import_module("gym_multi_car_racing")

run_dir = Path("experiments/single_sac_cnn_seed42_20260314_102245")
env_kwargs = {"continuous": True, "num_agents": 1, "render_mode": "human"}

# Build the SAME pre-normalize wrapper stack as training
eval_env = make_vec_env(
    "MultiCarRacing-v0",
    n_envs=1,
    vec_env_cls=DummyVecEnv,
    env_kwargs=env_kwargs,
)
eval_env = VecMonitor(eval_env)
eval_env = VecTransposeImage(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = SAC.load(str(run_dir / "best_model" / "best_model.zip"), env=eval_env, device="mps")

vec_env = model.get_env()
obs = vec_env.reset()
for _i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # Call each underlying env's render directly (opens pygame window for human mode)
    vec_env.env_method("render")
