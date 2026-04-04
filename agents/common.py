from hydra.utils import instantiate


def build_sb3_env(env_cfg, wrappers):
    env = instantiate(env_cfg)
    for wrapper in wrappers:
        env = instantiate(wrapper, venv=env)
    return env


def build_ss_env(env_cfg, wrappers):
    env_kwargs = env_cfg.pop("env_kwargs", {})
    env = instantiate(env_cfg, **env_kwargs)
    for wrapper in wrappers:
        env = instantiate(wrapper, env)
    return env
