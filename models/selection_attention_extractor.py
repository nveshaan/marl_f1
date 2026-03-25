import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SpatialSelectionAttention(nn.Module):
    """
    Selection-style spatial attention.
    Learns an importance mask over HxW and reweights CNN features.
    """
    def __init__(self):
        super().__init__()
        # use both average and max pooling across channels as cues
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = x.mean(dim=1, keepdim=True)          # [B, 1, H, W]
        mx, _ = x.max(dim=1, keepdim=True)         # [B, 1, H, W]
        pooled = torch.cat([avg, mx], dim=1)       # [B, 2, H, W]
        mask = self.sigmoid(self.conv(pooled))     # [B, 1, H, W] in (0,1)
        return x * mask                            # elementwise selection

class CNNWithSelectionAttention(BaseFeaturesExtractor):
    """
    Reuses a standard SB3-style CNN, then applies SpatialSelectionAttention,
    then flattens to features_dim.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        # Standard NatureCNN architecture used by SB3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.attn = SpatialSelectionAttention()

        # Compute flatten size by doing a dummy forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample = sample / 255.0
            n_flatten = self.attn(self.cnn(sample)).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass un-normalized observations if SB3 hasn't normalized them yet
        x = observations / 255.0
        x = self.cnn(x)
        x = self.attn(x)
        return self.linear(x)

