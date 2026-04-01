from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


def build_env(env_cfg, wrappers):
    # Handle env_kwargs by restructuring them properly for make_vec_env
    env_cfg_dict = OmegaConf.to_container(env_cfg, resolve=True)
    env_kwargs = env_cfg_dict.pop('env_kwargs', None)
    
    # For make_vec_env, env_kwargs should stay as a separate dict, not merged
    target = env_cfg_dict.get('_target_', '')
    if 'make_vec_env' in target and env_kwargs:
        # make_vec_env accepts env_kwargs as a separate parameter
        env_cfg_dict['env_kwargs'] = env_kwargs
    elif env_kwargs:
        # For other targets, merge env_kwargs into the main config
        env_cfg_dict.update(env_kwargs)
    
    # Create a new config from the merged dict
    env_cfg_merged = OmegaConf.create(env_cfg_dict)
    
    env = instantiate(env_cfg_merged)
    
    # Map wrapper class names to their expected environment parameter names
    wrapper_param_names = {
        'dtype_v0': 'env',
        'normalize_obs_v0': 'env',
        'frame_stack_v1': 'env',
        'pettingzoo_env_to_vec_env_v1': 'parallel_env',
        'concat_vec_envs_v1': 'vec_env',
        'VecMonitor': 'venv',
        'VecFrameStack': 'venv',
        'VecTransposeImage': 'venv',
        'VecNormalize': 'venv',
    }
    
    for wrapper in wrappers:
        # Extract wrapper target from DictConfig
        wrapper_target = ''
        if isinstance(wrapper, DictConfig):
            wrapper_target = wrapper.get('_target_', '')
        elif isinstance(wrapper, dict):
            wrapper_target = wrapper.get('_target_', '')
        
        wrapper_name = wrapper_target.split('.')[-1] if wrapper_target else ''
        
        # Determine the parameter name to use
        param_name = wrapper_param_names.get(wrapper_name, 'env')  # Default to 'env'
        
        # Instantiate the wrapper with the correct parameter name
        env = instantiate(wrapper, **{param_name: env})
    
    # Wrap the environment with SB3's DummyVecEnv if it's a supersuit environment
    # This ensures compatibility with SB3 algorithms like DQN
    env_type_name = type(env).__name__
    if 'supersuit' in str(type(env).__module__) or 'MarkovVectorEnv' in env_type_name:
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            # Create a wrapper class that returns the environment
            class EnvWrapper:
                def __init__(self, env):
                    self.env = env
                def __call__(self):
                    return self.env
            # Wrap single env: DummyVecEnv expects a list of callables
            env = DummyVecEnv([EnvWrapper(env)])
        except Exception as e:
            print(f"Warning: Could not wrap environment: {e}")
            pass
    
    # Add a seed method if the environment doesn't have one
    # This is needed for compatibility with SB3 algorithms
    if not hasattr(env, 'seed') or not callable(getattr(env, 'seed', None)):
        env.seed = lambda seed=None: [seed]
    
    return env
