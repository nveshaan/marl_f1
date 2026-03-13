from hydra.utils import instantiate
from stable_baselines3.common.callbacks import CallbackList


def build_env(env_cfg, wrappers):
    env = instantiate(env_cfg)
    for wrapper in wrappers:
        env = instantiate(wrapper, venv=env)
    return env


def build_callbacks(cfg, eval_env=None, run_dir=None, **kwargs):
    callbacks = []
    for callback in cfg:
        if not callback.enabled:
            continue
        instantiate_kwargs = {"_recursive_": False}
        if "eval_env" in callback.params:
            instantiate_kwargs["eval_env"] = eval_env
        callbacks.append(instantiate(callback.params, **instantiate_kwargs))
    return CallbackList(callbacks) if callbacks else None
