import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .selection_attention_extractor import SpatialSelectionAttention


class CTDEFeaturesExtractor(BaseFeaturesExtractor):
    """Efficient CNN extractor for stacked CTDE image observations.

    This version processes the full tensor in one batched CNN pass instead of
    looping over agents in Python.
    """

    def __init__(self, observation_space, features_dim=512, n_agents=None):
        super().__init__(observation_space, features_dim)
        self.n_channels = observation_space.shape[0] // n_agents
        self.n_agents = n_agents

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()  # (1, C, H, W)
            sample = sample[:, : self.n_channels] / 255.0
            n_flatten = self.cnn(sample).flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _group_agent_channels(self, observations):
        """Regroup channels to per-agent tensors in one vectorized pass.

        Expected channel order in input is frame-major with agent RGB blocks:
        [f0_a0_rgb, f0_a1_rgb, ..., f1_a0_rgb, f1_a1_rgb, ...].
        """
        single_image_channels = 3
        b, c, h, w = observations.shape
        frame_block = self.n_agents * single_image_channels

        if c % frame_block != 0:
            raise ValueError(
                f"Invalid channel count {c}: expected multiple of {frame_block} "
                f"(n_agents={self.n_agents}, rgb={single_image_channels})."
            )

        n_frames = c // frame_block
        if n_frames * single_image_channels != self.n_channels:
            raise ValueError(
                f"Per-agent channels mismatch: got {self.n_channels}, expected "
                f"{n_frames * single_image_channels}."
            )

        # (B, C, H, W) -> (B, F, N, 3, H, W) -> (B, N, F*3, H, W)
        observations = observations.reshape(b, n_frames, self.n_agents, single_image_channels, h, w)
        observations = observations.permute(0, 2, 1, 3, 4, 5)
        return observations.reshape(b, self.n_agents, self.n_channels, h, w)

    def forward(self, observations):
        observations = observations.float()
        observations = observations / 255.0
        observations = self._group_agent_channels(observations)  # B, N, C_per_agent, H, W
        x = observations.reshape(
            -1, self.n_channels, *observations.shape[3:]
        )  # (B*N), C_per_agent, H, W
        x = self.cnn(x)
        x = self.linear(x)
        return x.view(-1, self.n_agents, x.shape[1])  # B, (N*features_dim)


class CTDEFeaturesExtractorWithSelectionAttention(CTDEFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, n_agents=None):
        super().__init__(observation_space, features_dim, n_agents)
        self.attn = SpatialSelectionAttention()

    def forward(self, observations):
        observations = observations.float()
        observations = observations / 255.0
        observations = self._group_agent_channels(observations)  # B, N, C_per_agent, H, W
        observations = observations.view(
            -1, self.n_channels, *observations.shape[3:]
        )  # (B*N), C_per_agent, H, W
        x = self.cnn(observations)
        x = self.attn(x)
        x = self.linear(x)
        return x.view(-1, self.n_agents, x.shape[1])  # B, (N*features_dim)
