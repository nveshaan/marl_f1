from collections.abc import Callable

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import Actor, ContinuousCritic, TD3Policy


def _local_actor_features(features: th.Tensor) -> th.Tensor:
    """Use only ego-agent features for decentralized actor inference."""
    if features.ndim == 3:
        return features[:, 0]
    return features


def _central_critic_features(features: th.Tensor) -> th.Tensor:
    """Flatten all agents' features for centralized critic input."""
    if features.ndim == 3:
        return features.flatten(start_dim=1)
    return features


class MADDPGActor(Actor):
    """Actor that follows CTDE by acting from local (ego) features only."""

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(_local_actor_features(features))


class MADDPGCritic(ContinuousCritic):
    """Critic that consumes centralized features from all agents."""

    def __init__(
        self, *args, features_extractor: BaseFeaturesExtractor, features_dim: int, **kwargs
    ):
        n_agents = getattr(features_extractor, "n_agents", 1)
        centralized_features_dim = features_dim * n_agents
        super().__init__(
            *args,
            features_extractor=features_extractor,
            features_dim=centralized_features_dim,
            **kwargs,
        )

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # Mirror SB3's gradient handling when actor/critic share an extractor.
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([_central_critic_features(features), actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([_central_critic_features(features), actions], dim=1)
        return self.q_networks[0](qvalue_input)


class MADDPG(TD3Policy):
    """DDPG/TD3 policy with MAPPO-style CTDE actor/critic feature routing."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> MADDPGActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MADDPGActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> MADDPGCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return MADDPGCritic(**critic_kwargs).to(self.device)
