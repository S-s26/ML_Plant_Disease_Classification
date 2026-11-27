"""Neural Network (MLP) and classical ML models operating on handcrafted image features."""

import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """Multi-Layer Perceptron trained on standardized handcrafted image features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for idx, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            drop_rate = dropout if idx == 0 else max(0.15, dropout - 0.1 * idx)
            layers.append(nn.Dropout(drop_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
