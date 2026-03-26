from pathlib import Path

from hydra.utils import instantiate
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .base_agent import BaseAgent


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
        algo_cls = instantiate(self.cfg.algo.model, _partial_=True)
        self.model = algo_cls.load(path, env=self.train_env)

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
