import importlib
import re
from pathlib import Path
import datetime

import numpy as np
import torch
import cv2

from stable_baselines3 import SAC, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)

# ==============================
# REGISTER ENV
# ==============================
importlib.import_module("multi_car_racing")

# ==============================
# SELECT RUN
# ==============================
def select_run(base_dir="experiments"):
    base_path = Path(base_dir)

    runs = [
        p for p in base_path.iterdir()
        if p.is_dir() and (p / "best_model").exists()
    ]

    print("\nAvailable runs:\n")
    for i, run in enumerate(runs):
        print(f"[{i}] {run.name}")

    idx = int(input("\nSelect run index: "))
    return runs[idx]

run_dir = select_run("experiments")
print(f"\nSelected run: {run_dir}\n")

# ==============================
# PARSE NAME
# format: task_algo_policy_seedX
# ==============================
name = run_dir.name

pattern = r"(.+?)_(.+?)_(.+?)_seed(\d+)"
match = re.match(pattern, name)

if not match:
    raise ValueError("Run name format incorrect")

task, algo_name, policy_name, seed = match.groups()
seed = int(seed)

print(f"Task: {task}, Algo: {algo_name}, Policy: {policy_name}, Seed: {seed}")

# ==============================
# LOAD MODEL
# ==============================
ALGO_MAP = {
    "sac": SAC,
    "ppo": PPO,
    "dqn": DQN,
}

algo_cls = ALGO_MAP.get(algo_name.lower())
if algo_cls is None:
    raise ValueError(f"Unsupported algo: {algo_name}")

# ==============================
# ENV
# ==============================
env_kwargs = {
    "continuous": True,
    "num_agents": 1,
    "render_mode": "rgb_array"
}

eval_env = make_vec_env(
    "MultiCarRacing-v2",
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
eval_env.norm_obs = False

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = algo_cls.load(
    str(run_dir / "best_model" / "best_model.zip"),
    env=eval_env,
    device=device
)

# ==============================
# SELECT HOOK TARGET
# ==============================
def get_extractor(model):
    policy = model.policy

    if hasattr(policy, "actor"):
        return policy.actor.features_extractor
    elif hasattr(policy, "q_net"):
        return policy.q_net.features_extractor
    else:
        return policy.features_extractor

extractor = get_extractor(model)

# ==============================
# HOOK TYPE BASED ON POLICY
# ==============================
class FeatureHook:
    def __init__(self, extractor, policy_name):
        self.features = {}
        self.handles = []

        for name, layer in extractor.named_modules():

            # CNN policy → hook ReLU
            if "cnn" in policy_name.lower():
                if isinstance(layer, torch.nn.ReLU):
                    self.handles.append(layer.register_forward_hook(self._hook(name)))

            # Attention policy → hook attention layers
            elif "attn" in policy_name.lower():
                if "attn" in name.lower():
                    self.handles.append(layer.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            self.features[name] = out.detach().cpu()
        return fn

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()

# ==============================
# SAVE DIR
# ==============================
viz_dir = run_dir / "feature_viz"
obs_dir = viz_dir / "frames"
fmap_dir = viz_dir / "feature_maps"

obs_dir.mkdir(parents=True, exist_ok=True)
fmap_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# SAVE FUNCTIONS
# ==============================
def save_obs(vec_env, step):
    frame = vec_env.env_method("render")[0]

    if frame is None:
        return None

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(obs_dir / f"frame_{step:05d}.png"), frame_bgr)

    return frame


def save_feature_maps(feature_maps, step):
    for name, fmap in feature_maps.items():
        fmap = fmap.squeeze(0).cpu().numpy()

        np.save(str(fmap_dir / f"{name}_{step:05d}.npy"), fmap)

        # only handle spatial 3D feature maps (C, H, W); skip non-image tensors
        if fmap.ndim != 3:
            continue

        C = fmap.shape[0]

        for c in range(min(8, C)):
            fmap_img = fmap[c]

            min_val, max_val = fmap_img.min(), fmap_img.max()
            if max_val - min_val > 1e-8:
                fmap_img = (fmap_img - min_val) / (max_val - min_val)
            else:
                fmap_img = np.zeros_like(fmap_img)

            fmap_img = (fmap_img * 255).astype(np.uint8)

            # Ensure shape is HxW or HxWxC; cv2.imwrite requires numpy array
            if fmap_img.ndim == 2:
                pass
            elif fmap_img.ndim == 3 and fmap_img.shape[0] in (1, 3):
                fmap_img = np.transpose(fmap_img, (1, 2, 0))
            else:
                fmap_img = np.asarray(fmap_img)

            cv2.imwrite(str(fmap_dir / f"{name}_ch{c}_{step:05d}.png"), fmap_img)

# ==============================
# RUN
# ==============================
hook = FeatureHook(extractor, policy_name)

vec_env = model.get_env()
obs = vec_env.reset()

step = 0
# vec_env.step returns arrays for each env; with n_envs=1, done is a 1-element array.
while True:
    hook.clear()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    frame = save_obs(vec_env, step)
    save_feature_maps(hook.features, step)

    if frame is not None:
        cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    step += 1

    # In VecEnv, done is an array; stop when any env is done.
    if isinstance(done, (list, tuple, np.ndarray)):
        if any(done):
            break
    elif done:
        break

print(f"Done after {step} steps")

hook.remove()
cv2.destroyAllWindows()
