"""Inference engine supporting Transfer CNN, Custom CNN, Feature-MLP, and Grad-CAM."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.features.extractor import PlantLeafFeatureExtractor
from src.models.classical_and_mlp import FeatureMLP
from src.models.custom_cnn import PlantDiseaseCNN
from src.models.gradcam import GradCAM, overlay_gradcam
from src.models.transfer_cnn import TransferPlantCNN
from src.utils.io import load_config, load_json, resolve_path


class PlantDiseasePredictor:
    """Unified predictor for leaf images using CNNs + handcrafted image features."""

    def __init__(self, config_path: str | None = None) -> None:
        self.cfg = load_config(config_path)
        self.models_dir = resolve_path(self.cfg["paths"]["models_dir"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_map = load_json(self.models_dir / "class_mapping.json")
        self.class_names: List[str] = self.class_map["class_names"]
        self.healthy_ids: List[int] = self.class_map["healthy_class_ids"]

        self.image_size = int(self.cfg["dataset"]["image_size"])
        self.eval_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.extractor = PlantLeafFeatureExtractor(
            image_size=int(self.cfg["dataset"]["feature_image_size"]),
            lbp_radius=int(self.cfg["features"]["lbp_radius"]),
            lbp_points=int(self.cfg["features"]["lbp_points"]),
        )

        self.cnn_models: Dict[str, torch.nn.Module] = {}
        self._load_available_models()

    def _load_available_models(self) -> None:
        backbone = str(self.cfg["training"]["transfer_backbone"])
        transfer_ckpt = self.models_dir / f"transfer_{backbone}_multiclass.pt"
        if transfer_ckpt.exists():
            ckpt = torch.load(transfer_ckpt, map_location=self.device, weights_only=True)
            m = TransferPlantCNN(
                num_classes=int(ckpt["num_classes"]),
                backbone_name=ckpt["backbone_name"],
                pretrained=False,
            )
            m.load_state_dict(ckpt["state_dict"])
            self.cnn_models[f"Transfer CNN ({backbone})"] = m.to(self.device).eval()

        custom_ckpt = self.models_dir / "custom_cnn_multiclass.pt"
        if custom_ckpt.exists():
            ckpt = torch.load(custom_ckpt, map_location=self.device, weights_only=True)
            m = PlantDiseaseCNN(num_classes=int(ckpt["num_classes"]))
            m.load_state_dict(ckpt["state_dict"])
            self.cnn_models["Custom CNN (From Scratch)"] = m.to(self.device).eval()

        scaler_path = self.models_dir / "feature_scaler.joblib"
        mlp_path = self.models_dir / "feature_mlp_multiclass.pt"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.feature_mlp: FeatureMLP | None = None
        if mlp_path.exists():
            ckpt = torch.load(mlp_path, map_location=self.device, weights_only=True)
            mlp = FeatureMLP(
                in_features=int(ckpt["in_features"]),
                num_classes=int(ckpt["num_classes"]),
            )
            mlp.load_state_dict(ckpt["state_dict"])
            self.feature_mlp = mlp.to(self.device).eval()

    def available_models(self) -> List[str]:
        names = list(self.cnn_models.keys())
        if self.feature_mlp is not None and self.scaler is not None:
            names.append("Feature-MLP (Handcrafted Features)")
        return names

    def predict(
        self,
        image: Image.Image | str | Path,
        model_choice: str | None = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run end-to-end prediction, feature inspection, and Grad-CAM (for CNN models)."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        rgb_np = np.array(pil_img)
        feat_vec = self.extractor.extract_from_rgb(rgb_np)
        feat_dict = dict(zip(self.extractor.feature_names, feat_vec.tolist()))

        choices = self.available_models()
        if not choices:
            raise RuntimeError("No trained models found in models/ directory.")
        chosen = model_choice if model_choice in choices else choices[0]

        gradcam_overlay: np.ndarray | None = None
        if chosen in self.cnn_models:
            model = self.cnn_models[chosen]
            inp = self.eval_tf(pil_img).unsqueeze(0).to(self.device)
            cam = GradCAM(model, model.get_target_conv_layer())
            try:
                heatmap, _, _ = cam.generate(inp)
                rgb_resized = np.array(pil_img.resize((self.image_size, self.image_size)))
                gradcam_overlay = overlay_gradcam(rgb_resized, heatmap, alpha=0.45)
            finally:
                cam.remove_hooks()
            with torch.no_grad():
                probs = F.softmax(model(inp), dim=1).squeeze(0).cpu().numpy()
        else:
            assert self.scaler is not None and self.feature_mlp is not None
            scaled = self.scaler.transform(feat_vec.reshape(1, -1)).astype(np.float32)
            with torch.no_grad():
                logits = self.feature_mlp(torch.from_numpy(scaled).to(self.device))
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "class_id": int(idx),
                "class_name": self.class_names[int(idx)],
                "probability": float(probs[int(idx)]),
            }
            for idx in top_indices
        ]

        healthy_prob = float(probs[self.healthy_ids].sum())
        diseased_prob = float(np.clip(1.0 - healthy_prob, 0.0, 1.0))
        binary_verdict = "Healthy" if healthy_prob >= 0.5 else "Diseased"

        key_indicators = {
            "Excess Green Healthy Ratio": round(feat_dict["veg_exg_green_ratio"], 4),
            "Yellow/Brown Chlorosis Ratio": round(feat_dict["veg_yellow_brown_ratio"], 4),
            "Dark Necrotic Spot Ratio": round(feat_dict["veg_dark_necrotic_ratio"], 4),
            "CIELAB a* (Red-Green Shift)": round(feat_dict["lab_a_mean"], 4),
            "GLCM Texture Contrast (d=1)": round(feat_dict["glcm_d1_contrast"], 4),
            "Canny Edge Density": round(feat_dict["edge_canny_ratio"], 4),
        }

        return {
            "model_used": chosen,
            "binary_verdict": binary_verdict,
            "healthy_probability": healthy_prob,
            "diseased_probability": diseased_prob,
            "top_class": top_predictions[0]["class_name"],
            "top_confidence": top_predictions[0]["probability"],
            "top_predictions": top_predictions,
            "key_image_features": key_indicators,
            "gradcam_overlay": gradcam_overlay,
        }
