import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MAPPOFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for MAPPO that handles stacked CTDE observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, num_agents: int = 2):
        super().__init__(observation_space, features_dim)
        self.num_agents = num_agents
        channels = observation_space.shape[-1]  # Last dim is channels
        height, width = observation_space.shape[1], observation_space.shape[2]

        # CNN for processing the stacked observations
        # Treat the stack as a larger image: (num_agents * height, width, channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output dim
        with torch.no_grad():
            sample_input = torch.zeros(1, channels, num_agents * height, width)
            sample_output = self.cnn(sample_input)
            self._features_dim = sample_output.shape[1]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, num_agents, height, width, channels)
        batch_size = observations.shape[0]
        num_agents = observations.shape[1]
        height, width, channels = (
            observations.shape[2],
            observations.shape[3],
            observations.shape[4],
        )

        # Reshape to (batch, channels, num_agents * height, width)
        obs_reshaped = observations.permute(0, 4, 1, 2, 3).reshape(
            batch_size, channels, num_agents * height, width
        )

        return self.cnn(obs_reshaped)


class MAPPOPolicy(ActorCriticPolicy):
    """Custom policy for MAPPO that uses the stacked CTDE observations."""

    def __init__(self, observation_space, action_space, lr_schedule, num_agents=2, **kwargs):
        self.num_agents = num_agents
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MAPPOFeaturesExtractor,
            features_extractor_kwargs={"num_agents": num_agents},
            **kwargs,
        )

    def forward(self, obs, deterministic=False):
        # obs: (batch, num_agents, height, width, channels)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Actor: output action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Critic: value from features
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        values = self.value_net(latent_vf)

        return values, log_prob, entropy


class MAPPO(PPO):
    """MAPPO implementation inheriting from PPO with custom policy for CTDE."""

    def __init__(self, policy, env, num_agents=2, **kwargs):
        # Set custom policy if not provided
        if policy == "MlpPolicy" or policy is None:
            policy = MAPPOPolicy
            kwargs["policy_kwargs"] = kwargs.get("policy_kwargs", {})
            kwargs["policy_kwargs"]["num_agents"] = num_agents

        super().__init__(policy, env, **kwargs)
        self.num_agents = num_agents


if __name__ == "__main__":
    """Test MAPPO with CTDE PettingZoo environment."""
    import supersuit as ss

    from multi_car_racing import MultiCarRacingParallelEnv

    print("Creating CTDE environment...")
    env = MultiCarRacingParallelEnv(num_agents=2, ctde=True, include_actions=False)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.dtype_v0(env, "float32")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=2, num_cpus=1, base_class="stable_baselines3")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("Creating MAPPO model...")
    model = MAPPO(
        MAPPOPolicy, env, num_agents=2, verbose=1, n_steps=128, batch_size=256, n_epochs=4
    )

    print("Training for a few steps...")
    model.learn(total_timesteps=1000, progress_bar=True)

    print("Testing prediction...")
    obs = env.reset()[0]
    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action shape: {action.shape}")

    print("Test completed successfully!")
