"""Custom feature extractor for Splendor observations.

A 4-layer MLP that maps the 203-dim flat observation to a 256-dim embedding.
Uses LayerNorm for training stability and orthogonal initialisation (SB3 convention).

Hardware: ~760K params, ~4 MB GPU memory.
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SplendorFeatureExtractor(BaseFeaturesExtractor):
    """MLP feature extractor for Splendor's 203-dim observation vector.

    Architecture: 203 -> 512 -> LN -> ReLU -> 512 -> LN -> ReLU
                       -> 512 -> LN -> ReLU -> 256 -> ReLU
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]  # 203

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation with sqrt(2) gain (SB3 standard)."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
