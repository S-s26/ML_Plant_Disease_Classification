"""Custom Convolutional Neural Network built from scratch for leaf disease classification."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double 3x3 convolution block with BatchNorm, ReLU, and 2x2 MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, dropout_2d: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_2d > 0.0:
            layers.append(nn.Dropout2d(dropout_2d))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlantDiseaseCNN(nn.Module):
    """4-stage CNN with Global Average Pooling built from scratch (~1.2M parameters)."""

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.35) -> None:
        super().__init__()
        self.stage1 = ConvBlock(in_channels, 32, dropout_2d=0.0)
        self.stage2 = ConvBlock(32, 64, dropout_2d=0.05)
        self.stage3 = ConvBlock(64, 128, dropout_2d=0.10)
        self.stage4 = ConvBlock(128, 256, dropout_2d=0.10)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(128, num_classes),
        )

    def get_target_conv_layer(self) -> nn.Module:
        """Return final convolutional layer prior to final pool for Grad-CAM."""
        # stage4.block indices: 0=Conv,1=BN,2=ReLU,3=Conv,4=BN,5=ReLU,6=Drop2d,7=Pool
        return self.stage4.block[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)
