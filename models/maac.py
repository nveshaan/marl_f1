from collections.abc import Callable

import torch as th
from gymnasium import spaces
from torch import nn

from .mappo import MAPPO


class CriticCrossAttentionNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_agents: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        n_heads: int = 2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        if n_agents < 1:
            raise ValueError(f"n_agents must be positive, got {n_agents}")
        if feature_dim % n_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.policy_net = nn.Sequential(nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU())
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.value_net = nn.Sequential(nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU())

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features[:, 0]), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected critic features with shape [B, N, D], got {features.shape}")

        ego_features = features[:, :1]
        other_features = features[:, 1:]

        if other_features.shape[1] == 0:
            attended_features = ego_features.squeeze(1)
        else:
            attended_features, _ = self.attn(
                query=ego_features, key=other_features, value=other_features
            )
            attended_features = attended_features.squeeze(1)

        return self.value_net(attended_features)


class MAAC(MAPPO):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        critic_n_heads: int = 4,
        *args,
        **kwargs,
    ):
        self.critic_n_heads = critic_n_heads
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CriticCrossAttentionNet(
            self.features_dim,
            n_agents=self.features_extractor.n_agents,
            n_heads=self.critic_n_heads,
        )
