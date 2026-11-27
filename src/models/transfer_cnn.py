"""Transfer Learning CNN backbones (MobileNetV2 / ResNet18) for plant leaf classification."""

import torch
import torch.nn as nn
from torchvision import models


class TransferPlantCNN(nn.Module):
    """Pretrained torchvision backbone with custom classification head."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v2",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = backbone
        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = backbone
        else:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose 'mobilenet_v2' or 'resnet18'.")

    def get_target_conv_layer(self) -> nn.Module:
        """Return final convolutional feature map layer for Grad-CAM."""
        if self.backbone_name == "mobilenet_v2":
            return self.model.features[-1]
        return self.model.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
