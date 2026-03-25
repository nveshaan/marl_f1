from pathlib import Path
import importlib

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .base_agent import BaseAgent


class SingleAgent(BaseAgent):
    """SB3 agent."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.run_dir = Path(cfg.run_dir)
        self.train_env = instantiate(cfg.agent.train_env, _recursive_=False)
        self.eval_env = instantiate(cfg.agent.eval_env, _recursive_=False)

        # ── NEW: resolve policy_kwargs and convert features_extractor_class str → class ──
        policy_kwargs = OmegaConf.to_container(cfg.policy.policy_kwargs, resolve=True)
        if isinstance(policy_kwargs, dict) and "features_extractor_class" in policy_kwargs:
            fec = policy_kwargs["features_extractor_class"]
            if isinstance(fec, str):
                module_path, class_name = fec.rsplit(".", 1)
                policy_kwargs["features_extractor_class"] = getattr(
                    importlib.import_module(module_path), class_name
                )
        # ────────────────────────────────────────────────────────────────────────────────

        self.model = instantiate(
            cfg.algo.model,
            env=self.train_env,
            policy_kwargs=policy_kwargs or None,
        )

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
        self.model = algo_cls.load(path, env=self.train_env, device=self.cfg.device)

    def save(self, path: str) -> None:
        self.model.save(str(path / "final_model"))
        self.train_env.save(str(path / "vecnormalize.pkl"))

    def eval(self) -> None:
        raise NotImplementedError

    def get_model(self):
        return self.model

    def get_env(self):
        return self.train_env


class MultiAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

