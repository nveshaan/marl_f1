import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .selection_attention_extractor import SpatialSelectionAttention


# TODO: look into how to allow dynamic number of agents
class CTDEFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.n_agents = len(observation_space["global_obs"].spaces)
        n_channels = (
            observation_space["global_obs"]["agent_0"].shape[-1] * self.n_agents
        )  # assuming all agents have same obs shape

        # Standard NatureCNN architecture used by SB3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute flatten size by doing a dummy forward pass
        with torch.no_grad():
            sample = observation_space.sample()
            sample_tensor = torch.as_tensor(sample["global_obs"]["agent_0"][None]).float()
            sample_tensor = sample_tensor.reshape(1, -1, 96, 96) / 255.0  # Proper shape
            n_flatten = self.cnn(sample_tensor).view(1, -1).shape[1]

        if "global_actions" in observation_space:
            action_size = observation_space
            n_flatten += action_size

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # observations["global_obs"] = dict of frames
        # Stack all frames and process through CNN
        frames = [
            observations["global_obs"][f"agent_{i}"][:: self.n_agents] for i in range(self.n_agents)
        ]
        if "global_actions" in observations:
            actions = [
                observations["global_actions"][f"agent_{i}"][:: self.n_agents]
                for i in range(self.n_agents)
            ]
        x = torch.stack(frames, dim=1)  # (batch, agents, channels, h, w)
        x = x.view(
            x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]
        )  # (batch, agents*channels, h, w)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)  # flatten
        actions_flat = (
            torch.cat([a.flatten(1) for a in actions], dim=1)
            if "global_actions" in observations
            else None
        )
        final = torch.cat([x, actions_flat], dim=1) if actions_flat is not None else x
        return self.linear(final)


class CTDEFeaturesExtractorWithSelectionAttention(CTDEFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.attn = SpatialSelectionAttention()

    def forward(self, observations):
        # observations["global_obs"] = dict of frames
        # Stack all frames and process through CNN
        frames = [
            observations["global_obs"][f"agent_{i}"][:: self.n_agents] for i in range(self.n_agents)
        ]
        if "global_actions" in observations:
            actions = [
                observations["global_actions"][f"agent_{i}"][:: self.n_agents]
                for i in range(self.n_agents)
            ]
        x = torch.stack(frames, dim=1)  # (batch, agents, channels, h, w)
        x = x.view(
            x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]
        )  # (batch, agents*channels, h, w)
        x = self.cnn(x)
        x = self.attn(x)
        x = x.view(x.shape[0], -1)  # flatten
        actions_flat = (
            torch.cat([a.flatten(1) for a in actions], dim=1)
            if "global_actions" in observations
            else None
        )
        final = torch.cat([x, actions_flat], dim=1) if actions_flat is not None else x
        return self.linear(final)
