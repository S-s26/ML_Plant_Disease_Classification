"""Unit tests for FeatureMLP, Custom CNN, Transfer CNN, and Grad-CAM."""

import torch

from src.models.classical_and_mlp import FeatureMLP
from src.models.custom_cnn import PlantDiseaseCNN
from src.models.gradcam import GradCAM
from src.models.transfer_cnn import TransferPlantCNN


def test_feature_mlp_forward() -> None:
    model = FeatureMLP(in_features=114, num_classes=38)
    x = torch.randn(8, 114)
    out = model(x)
    assert out.shape == (8, 38)


def test_custom_cnn_forward_and_gradcam() -> None:
    model = PlantDiseaseCNN(num_classes=38)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 38)

    cam = GradCAM(model, model.get_target_conv_layer())
    try:
        hm, pred_idx, conf = cam.generate(x[:1])
        assert hm.shape == (224, 224)
        assert 0 <= pred_idx < 38
        assert 0.0 <= conf <= 1.0
    finally:
        cam.remove_hooks()


def test_transfer_cnn_forward_unpretrained() -> None:
    model = TransferPlantCNN(num_classes=2, backbone_name="mobilenet_v2", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
