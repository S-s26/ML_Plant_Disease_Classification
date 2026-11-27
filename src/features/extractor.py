"""Handcrafted color, vegetation-index, texture (GLCM, LBP), and shape/edge feature extractor."""

from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


class PlantLeafFeatureExtractor:
    """Extracts an interpretable 114-dimensional feature vector from a leaf RGB/BGR image."""

    def __init__(
        self,
        image_size: int = 128,
        hsv_bins: Tuple[int, int, int] = (8, 8, 8),
        lab_bins: Tuple[int, int, int] = (8, 8, 8),
        lbp_radius: int = 2,
        lbp_points: int = 16,
        glcm_distances: Tuple[int, ...] = (1, 3),
        glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    ) -> None:
        self.image_size = image_size
        self.hsv_bins = hsv_bins
        self.lab_bins = lab_bins
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.glcm_distances = tuple(glcm_distances)
        self.glcm_angles = tuple(glcm_angles)
        self._feature_names = self._build_feature_names()

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def num_features(self) -> int:
        return len(self._feature_names)

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        for ch in ("r", "g", "b"):
            names.extend([f"rgb_{ch}_mean", f"rgb_{ch}_std"])
        for ch, bins in zip(("h", "s", "v"), self.hsv_bins):
            names.extend([f"hsv_{ch}_mean", f"hsv_{ch}_std"])
            names.extend([f"hsv_{ch}_hist_{i}" for i in range(bins)])
        for ch, bins in zip(("l", "a", "b"), self.lab_bins):
            names.extend([f"lab_{ch}_mean", f"lab_{ch}_std"])
            names.extend([f"lab_{ch}_hist_{i}" for i in range(bins)])
        names.extend(
            [
                "veg_exg_mean",
                "veg_exg_std",
                "veg_exg_green_ratio",
                "veg_grvi_mean",
                "veg_grvi_std",
                "veg_yellow_brown_ratio",
                "veg_dark_necrotic_ratio",
                "veg_lab_a_positive_ratio",
            ]
        )
        for d in self.glcm_distances:
            for prop in GLCM_PROPERTIES:
                names.append(f"glcm_d{d}_{prop}")
        n_lbp_bins = self.lbp_points + 2
        names.extend([f"lbp_u_bin_{i}" for i in range(n_lbp_bins)])
        names.extend([f"hu_moment_{i}" for i in range(7)])
        names.extend(["grad_sobel_mean", "grad_sobel_std", "edge_canny_ratio"])
        return names

    @staticmethod
    def _channel_hist(channel: np.ndarray, bins: int, max_val: float) -> np.ndarray:
        hist = cv2.calcHist([channel], [0], None, [bins], [0, max_val]).flatten()
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist.astype(np.float32)

    def extract_from_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """Extract feature vector from an HxWx3 uint8 RGB numpy array."""
        if rgb_image is None or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image array.")

        resized = cv2.resize(
            rgb_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        feats: List[float] = []

        # 1. RGB normalized channel stats
        rgb_norm = resized.astype(np.float32) / 255.0
        for idx in range(3):
            ch = rgb_norm[:, :, idx]
            feats.extend([float(ch.mean()), float(ch.std())])

        # 2. HSV stats & marginal histograms
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        hsv_ranges = (180.0, 256.0, 256.0)
        for idx, (bins, max_v) in enumerate(zip(self.hsv_bins, hsv_ranges)):
            ch = hsv[:, :, idx]
            feats.extend([float(ch.mean() / max_v), float(ch.std() / max_v)])
            feats.extend(self._channel_hist(ch, bins, max_v).tolist())

        # 3. CIELAB stats & marginal histograms
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        for idx, bins in enumerate(self.lab_bins):
            ch = lab[:, :, idx]
            feats.extend([float(ch.mean() / 255.0), float(ch.std() / 255.0)])
            feats.extend(self._channel_hist(ch, bins, 256.0).tolist())

        # 4. Vegetation & Disease Color Indices
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        exg = 2.0 * g - r - b
        grvi = (g - r) / (g + r + 1e-5)

        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Yellow-to-orange/brown hues with moderate saturation & brightness (chlorosis/lesions)
        yellow_brown_mask = (h_ch >= 8) & (h_ch <= 34) & (s_ch >= 40) & (v_ch >= 30)
        # Very dark spots within saturated/non-background regions or general necrotic regions
        dark_necrotic_mask = (v_ch >= 15) & (v_ch <= 75) & (s_ch >= 30)
        # In OpenCV LAB, a* > 128 shifts toward red/magenta (non-green tissue)
        lab_a_pos = lab[:, :, 1] > 128

        feats.extend(
            [
                float(exg.mean()),
                float(exg.std()),
                float((exg > 0.05).mean()),
                float(grvi.mean()),
                float(grvi.std()),
                float(yellow_brown_mask.mean()),
                float(dark_necrotic_mask.mean()),
                float(lab_a_pos.mean()),
            ]
        )

        # 5. GLCM Texture Features (quantized to 64 levels for stable co-occurrence counts)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_q = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(
            gray_q,
            distances=list(self.glcm_distances),
            angles=list(self.glcm_angles),
            levels=64,
            symmetric=True,
            normed=True,
        )
        for d_idx in range(len(self.glcm_distances)):
            for prop in GLCM_PROPERTIES:
                val = graycoprops(glcm, prop)[d_idx, :].mean()
                feats.append(float(val))

        # 6. Uniform Local Binary Patterns (LBP)
        lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method="uniform")
        n_lbp_bins = self.lbp_points + 2
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_lbp_bins, range=(0, n_lbp_bins), density=True
        )
        feats.extend(np.nan_to_num(lbp_hist, nan=0.0).astype(np.float32).tolist())

        # 7. Hu Moments (log-signed scale for numerical stability)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        for val in hu:
            if val == 0.0:
                feats.append(0.0)
            else:
                feats.append(float(-np.sign(val) * np.log10(abs(val) + 1e-12)))

        # 8. Edge & Gradient Features
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2) / 255.0
        edges = cv2.Canny(gray, 50, 150)
        feats.extend(
            [
                float(grad_mag.mean()),
                float(grad_mag.std()),
                float((edges > 0).mean()),
            ]
        )

        vec = np.asarray(feats, dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_from_path(self, image_path: str | Path) -> np.ndarray:
        """Load an image file from disk in RGB order and extract features."""
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.extract_from_rgb(rgb)
