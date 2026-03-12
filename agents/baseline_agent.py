from pathlib import Path

from base_agent import BaseAgent
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList


class BaselineAgent(BaseAgent):
    """Model-Free Agent to act as the baseline against Model-Based agents.
    Can be used as a Single, Cooperative, Competitive, Mixed Agents setup.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.run_dir = Path(cfg.run_dir)
        self.train_env = self._build_env(cfg.train_env, is_eval=False)
        self.eval_env = self._build_env(cfg.eval_env, is_eval=True)
        self.model = self._build_model()

    def _build_env(self, env_cfg, is_eval: bool):
        env = instantiate(env_cfg)
        wrappers = self.cfg.wrappers.eval.wrappers if is_eval else self.cfg.wrappers.train.wrappers
        for wrapper in wrappers:
            env = instantiate(wrapper, venv=env)

        return env

    def _build_model(self):
        algo_cfg = {"_target_": self.cfg.algo._target_, **dict(self.cfg.algo.hyperparams)}
        return instantiate(OmegaConf.create(algo_cfg), env=self.train_env)

    def _build_callbacks(self):
        callbacks = []
        if self.cfg.callbacks.eval.enabled:
            eval_cfg = {
                "_target_": "stable_baselines3.common.callbacks.EvalCallback",
                "eval_freq": self.cfg.callbacks.eval.eval_freq,
                "n_eval_episodes": self.cfg.callbacks.eval.n_eval_episodes,
                "deterministic": self.cfg.callbacks.eval.deterministic,
            }
            callbacks.append(
                instantiate(
                    OmegaConf.create(eval_cfg),
                    eval_env=self.eval_env,
                    best_model_save_path=str(
                        self.run_dir / self.cfg.callbacks.eval.best_model_subdir
                    ),
                    log_path=str(self.run_dir / self.cfg.callbacks.eval.log_subdir),
                )
            )
        if self.cfg.callbacks.checkpoint.enabled:
            checkpoint_cfg = {
                "_target_": "stable_baselines3.common.callbacks.CheckpointCallback",
                "save_freq": self.cfg.callbacks.checkpoint.save_freq,
                "name_prefix": self.cfg.callbacks.checkpoint.name_prefix,
            }
            callbacks.append(
                instantiate(
                    OmegaConf.create(checkpoint_cfg),
                    save_path=str(self.run_dir / self.cfg.callbacks.checkpoint.save_subdir),
                )
            )
        return CallbackList(callbacks) if callbacks else None

    def learn(self) -> None:
        callbacks = self._build_callbacks()
        self.model.learn(
            total_timesteps=self.cfg.train.total_timesteps,
            callback=callbacks,
            tb_log_name=self.cfg.train.tb_log_name,
        )
        self.save(str(self.run_dir / "final_model"))
        self.train_env.save(str(self.run_dir / "vecnormalize.pkl"))

    def load(self, path: str) -> None:
        algo_cls = get_class(self.cfg.algo._target_)
        self.model = algo_cls.load(path, env=self.train_env, device=self.cfg.runtime.device)

    def save(self, path: str) -> None:
        self.model.save(path)

    def eval(self) -> None:
        self.eval_env.training = False
        self.eval_env.norm_reward = False

    def get_model(self):
        return self.model

    def get_env(self):
        return self.train_env


def make_baseline_agent(cfg):
    return BaselineAgent(cfg)
