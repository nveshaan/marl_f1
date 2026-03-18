from pathlib import Path

from hydra.utils import get_class, instantiate
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .base_agent import BaseAgent


class BaselineAgent(BaseAgent):
    """Model-Free Agent to act as the baseline against Model-Based agents.
    Can be used as a Single, Cooperative, Competitive, Mixed Agents setup.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.run_dir = Path(cfg.run_dir)
        self.train_env = instantiate(cfg.train_env, _recursive_=False)
        self.eval_env = instantiate(cfg.eval_env, _recursive_=False)
        self.model = instantiate(cfg.algo.model, env=self.train_env)

        eval_cb = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=str(self.cfg.run_dir / "best_model"),
            log_path=str(self.cfg.run_dir / "eval_logs"),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=25_000,
            save_path=str(self.cfg.run_dir / "checkpoints"),
            name_prefix=self.cfg.algo.name,
        )
        self.callbacks = CallbackList([eval_cb, ckpt_cb])

    def learn(self) -> None:
        self.model.learn(callbacks=self.callbacks, **self.cfg.train)

    def load(self, path: str) -> None:
        algo_cls = get_class(self.cfg.algo.model._target_)
        self.model = algo_cls.load(path, env=self.train_env, device=self.cfg.device)

    def save(self, path: str) -> None:
        self.model.save(str(path / "final_model"))
        self.train_env.save(str(path / "vecnormalize.pkl"))

    def eval(self) -> None:  # FIXME
        self.eval_env.training = False
        self.eval_env.norm_reward = False

    def get_model(self):
        return self.model

    def get_env(self):
        return self.train_env


def make_baseline_agent(cfg):
    return BaselineAgent(cfg)
