import fnmatch
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .base_agent import BaseAgent


class FeatureHook:
    """Hook for capturing feature maps from specific layers."""

    def __init__(self, extractor, policy_name, selected_layers):
        """Initialize hooks on selected layers.

        Args:
            extractor: Feature extractor module.
            policy_name: Name of policy (cnn, attn, etc.)
            selected_layers: List of layer names to hook. If empty, hooks all applicable layers.
        """
        self.features = {}
        self.handles = []
        self.selected_layers = set(selected_layers) if selected_layers else None

        for name, layer in extractor.named_modules():
            if self.selected_layers is not None and name not in self.selected_layers:
                continue

            if (
                "cnn" in policy_name.lower()
                and isinstance(layer, torch.nn.ReLU)
                or "attn" in policy_name.lower()
                and ("attn" in name.lower() or isinstance(layer, torch.nn.MultiheadAttention))
            ):
                self.handles.append(layer.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        """Create a hook function that saves outputs."""

        def fn(module, inp, out):
            self.features[name] = out.detach().cpu()

        return fn

    def clear(self):
        """Clear captured features."""
        self.features = {}

    def remove(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()


class SingleAgent(BaseAgent):
    """SB3 single agent implementation. This agent is used for training and evaluating single agents.

    It is initialized with a configuration file that specifies the training and evaluation environments, the model architecture, and the training parameters. The agent can be trained, evaluated, and saved to disk.

    An example configuration file for this agent might look like this:
    agent:
        train_env:
            _target_: my_envs.MyTrainEnv
        eval_env:
            _target_: my_envs.MyEvalEnv
    algo:
        name: "PPO"
        model:
            _target_: stable_baselines3.PPO
            n_steps: 2048
            batch_size: 64
            n_epochs: 10
    policy:
        name: cnn
        policy: CnnPolicy
        policy_kwargs: {}
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.run_dir = Path(cfg.run_dir)
        self.policy_name = cfg.get("policy", {}).get("name", "unknown")
        self.train_env = instantiate(cfg.agent.train_env, _recursive_=False)
        self.eval_env = instantiate(cfg.agent.eval_env, _recursive_=False)
        self.model = instantiate(cfg.algo.model, env=self.train_env)

        eval_cb = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=str(self.run_dir / "best_model"),
            log_path=str(self.run_dir / "eval_logs"),
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=100000,
            save_path=str(self.run_dir / "checkpoints"),
            name_prefix=self.cfg.algo.name,
        )
        self.callbacks = CallbackList([eval_cb, ckpt_cb])

    def learn(self) -> None:
        self.model.learn(callback=self.callbacks, **self.cfg.agent.train)

    def load(self, path: str) -> None:
        algo_cls = get_class(self.cfg.algo.model._target_)
        # HACK: ideally we should resolve these before saving the config, but this is a quick fix to allow loading without errors
        model_kwargs = {
            k: v for k, v in self.cfg.algo.model.items() if k not in ["_target_", "policy_kwargs"]
        }
        self.model = algo_cls.load(path, env=self.train_env, **model_kwargs)

    def save(self, path: str) -> None:
        self.model.save(str(path / "final_model"))
        self.train_env.save(str(path / "vecnormalize.pkl"))

    def eval(self, num_steps=None, deterministic=True, selected_layers=None) -> None:
        """Run evaluation with feature visualization.

        Args:
            num_steps: Maximum number of steps to run. If None, runs until episode ends.
            deterministic: Whether to use deterministic policy.
            selected_layers: List of layer names to hook. If None, user will be prompted.
        """
        # Create output directories
        viz_dir = self.run_dir / "viz"
        obs_dir = viz_dir / "frames"
        fmap_dir = viz_dir / "feature_maps"
        obs_dir.mkdir(parents=True, exist_ok=True)
        fmap_dir.mkdir(parents=True, exist_ok=True)

        # Load best model
        best_model_path = self.run_dir / "best_model" / "best_model.zip"
        if best_model_path.exists():
            self.load(str(best_model_path))
        else:
            print(f"Warning: No best model found at {best_model_path}")

        # Get feature extractor
        extractor = self._get_feature_extractor()

        # Discover and prompt for layer selection if not provided
        if selected_layers is None:
            available_layers = self._discover_layers(extractor)
            selected_layers = self._select_layers(available_layers)

        # Setup hooks
        hook = FeatureHook(extractor, self.policy_name, selected_layers)

        # Run evaluation
        vec_env = self.model.get_env()
        obs = vec_env.reset()

        step = 0
        while True:
            hook.clear()

            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)

            self._save_obs(vec_env, obs_dir, step)
            self._save_feature_maps(hook.features, fmap_dir, step)

            # Visualize
            frame = vec_env.env_method("render")[0]
            if frame is not None:
                cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            step += 1

            # Check stopping conditions
            if num_steps is not None and step >= num_steps:
                break
            if isinstance(done, (list, tuple, np.ndarray)):
                if any(done):
                    break
            elif done:
                break

        print(f"Done after {step} steps")
        hook.remove()
        cv2.destroyAllWindows()

    def get_available_layers(self):
        """Get list of hookable layers after loading best model.

        Returns:
            List of (layer_name, layer) tuples.
        """
        best_model_path = self.run_dir / "best_model" / "best_model.zip"
        if best_model_path.exists():
            self.load(str(best_model_path))
        else:
            print(f"Warning: No best model found at {best_model_path}")

        extractor = self._get_feature_extractor()
        return self._discover_layers(extractor)

    def _get_feature_extractor(self):
        """Extract feature extractor from model."""
        policy = self.model.policy
        if hasattr(policy, "actor"):
            return policy.actor.features_extractor
        elif hasattr(policy, "q_net"):
            return policy.q_net.features_extractor
        else:
            return policy.features_extractor

    def _discover_layers(self, extractor):
        """Discover all hookable layers in feature extractor.

        Returns:
            List of (layer_name, layer) tuples.
        """
        layers = []

        for name, layer in extractor.named_modules():
            # CNN policy → ReLU layers
            if (
                "cnn" in self.policy_name.lower()
                and isinstance(layer, torch.nn.ReLU)
                or "attn" in self.policy_name.lower()
                and ("attn" in name.lower() or isinstance(layer, torch.nn.MultiheadAttention))
            ):
                layers.append((name, layer))

        return layers

    def _select_layers(self, available_layers):
        """Prompt user to select which layers to visualize.

        Args:
            available_layers: List of (layer_name, layer) tuples.

        Returns:
            List of layer names matching user selection.
        """
        if not available_layers:
            print("No hookable layers found")
            return []

        print("\n=== Available Layers ===")
        for i, (name, _) in enumerate(available_layers):
            print(f"[{i}] {name}")

        print("\nSelect layers to save (use patterns or indices):")
        print('  Examples: "all", "relu_*", "0,2,4", "attn_0,attn_2"')

        selection = input("\nYour selection: ").strip()

        if selection.lower() == "all":
            return [name for name, _ in available_layers]

        selected = []
        layer_names = [name for name, _ in available_layers]

        # Handle comma-separated indices or names
        for part in selection.split(","):
            part = part.strip()

            # Try parsing as index
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(layer_names):
                    selected.append(layer_names[idx])
            # Try pattern matching
            else:
                matched = fnmatch.filter(layer_names, part)
                selected.extend(matched)

        return list(set(selected))  # Remove duplicates

    def _save_obs(self, vec_env, obs_dir, step):
        """Save observation frame as image."""
        frame = vec_env.env_method("render")[0]
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(obs_dir / f"frame_{step:05d}.png"), frame_bgr)

    def _save_feature_maps(self, feature_maps, fmap_dir, step):
        """Save feature maps as .npy and visualizations."""
        for name, fmap in feature_maps.items():
            fmap = fmap.squeeze(0).cpu().numpy()

            np.save(str(fmap_dir / f"{name}_{step:05d}.npy"), fmap)

            # Handle spatial 3D feature maps (C, H, W)
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

                if fmap_img.ndim == 2:
                    pass
                elif fmap_img.ndim == 3 and fmap_img.shape[0] in (1, 3):
                    fmap_img = np.transpose(fmap_img, (1, 2, 0))
                else:
                    fmap_img = np.asarray(fmap_img)

                cv2.imwrite(str(fmap_dir / f"{name}_ch{c}_{step:05d}.png"), fmap_img)

    def get_model(self):
        return self.model

    def get_env(self):
        return self.train_env


class MultiAgent(SingleAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
