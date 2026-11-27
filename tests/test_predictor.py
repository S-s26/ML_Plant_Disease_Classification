"""End-to-end smoke test for PlantDiseasePredictor using temporary artifacts."""

from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from src.inference.predictor import PlantDiseasePredictor
from src.models.classical_and_mlp import FeatureMLP
from src.models.custom_cnn import PlantDiseaseCNN
from src.utils.io import save_json


def test_predictor_smoke_with_custom_cnn_and_mlp(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
project:
  name: test
  seed: 42
paths:
  models_dir: "{models_dir.as_posix()}"
dataset:
  image_size: 224
  feature_image_size: 128
features:
  lbp_radius: 2
  lbp_points: 16
training:
  transfer_backbone: mobilenet_v2
""",
        encoding="utf-8",
    )

    save_json(
        {
            "num_classes": 2,
            "class_names": ["Apple___healthy", "Apple___Apple_scab"],
            "class_to_id": {"Apple___healthy": 0, "Apple___Apple_scab": 1},
            "binary_classes": ["healthy", "diseased"],
            "binary_to_id": {"healthy": 0, "diseased": 1},
            "healthy_class_ids": [0],
        },
        models_dir / "class_mapping.json",
    )

    cnn = PlantDiseaseCNN(num_classes=2)
    torch.save(
        {"num_classes": 2, "state_dict": cnn.state_dict()},
        models_dir / "custom_cnn_multiclass.pt",
    )

    scaler = StandardScaler()
    scaler.fit(np.zeros((10, 114), dtype=np.float32))
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")

    mlp = FeatureMLP(in_features=114, num_classes=2)
    torch.save(
        {"in_features": 114, "num_classes": 2, "state_dict": mlp.state_dict()},
        models_dir / "feature_mlp_multiclass.pt",
    )

    predictor = PlantDiseasePredictor(config_path=str(cfg_path))
    assert "Custom CNN (From Scratch)" in predictor.available_models()
    assert "Feature-MLP (Handcrafted Features)" in predictor.available_models()

    dummy_img = Image.new("RGB", (256, 256), color=(40, 160, 50))
    res_cnn = predictor.predict(dummy_img, model_choice="Custom CNN (From Scratch)", top_k=2)
    assert res_cnn["binary_verdict"] in ("Healthy", "Diseased")
    assert res_cnn["gradcam_overlay"] is not None
    assert len(res_cnn["top_predictions"]) == 2

    res_mlp = predictor.predict(dummy_img, model_choice="Feature-MLP (Handcrafted Features)", top_k=2)
    assert res_mlp["gradcam_overlay"] is None
    assert len(res_mlp["key_image_features"]) == 6
