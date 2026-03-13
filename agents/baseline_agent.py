from pathlib import Path

from hydra.utils import get_class, instantiate

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
        self.callbacks = instantiate(cfg.callbacks, eval_env=self.eval_env, _recursive_=False)

    def learn(self) -> None:
        self.model.learn(
            total_timesteps=self.cfg.train.total_timesteps,
            callback=self.callbacks,
            tb_log_name=self.cfg.train.tb_log_name,
        )
        self.save(str(self.run_dir / "final_model"))
        self.train_env.save(str(self.run_dir / "vecnormalize.pkl"))

    def load(self, path: str) -> None:
        algo_cls = get_class(self.cfg.algo.model._target_)
        self.model = algo_cls.load(path, env=self.train_env, device=self.cfg.device)

    def save(self, path: str) -> None:
        self.model.save(path)

    def eval(self) -> None:  # FIXME
        self.eval_env.training = False
        self.eval_env.norm_reward = False

    def get_model(self):
        return self.model

    def get_env(self):
        return self.train_env


def make_baseline_agent(cfg):
    return BaselineAgent(cfg)
