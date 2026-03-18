from hydra.utils import instantiate


def build_env(env_cfg, wrappers):
    env = instantiate(env_cfg)
    for wrapper in wrappers:
        env = instantiate(wrapper, venv=env)
    return env
