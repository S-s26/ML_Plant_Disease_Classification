"""Grad-CAM explainability utility for CNN plant disease classifiers."""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Gradient-weighted Class Activation Maps for a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_inp: tuple, grad_out: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> Tuple[np.ndarray, int, float]:
        """Generate normalized [0, 1] heatmap, predicted class index, and confidence."""
        self.model.eval()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        cls_idx = pred_idx if target_class is None else int(target_class)
        confidence = float(probs[0, pred_idx].item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, cls_idx]
        score.backward()

        assert self._activations is not None and self._gradients is not None
        # Global-average-pool gradients across spatial dimensions -> channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np.astype(np.float32), pred_idx, confidence


def overlay_gradcam(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] Grad-CAM heatmap onto an HxWx3 uint8 RGB image."""
    h, w = rgb_image.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = np.uint8(255 * np.clip(hm_resized, 0.0, 1.0))
    bgr_jet = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    rgb_jet = cv2.cvtColor(bgr_jet, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * rgb_jet.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
