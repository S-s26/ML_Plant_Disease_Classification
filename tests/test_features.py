"""Unit tests for handcrafted image feature extraction."""

import numpy as np

from src.features.extractor import PlantLeafFeatureExtractor


def test_feature_vector_shape_and_finiteness() -> None:
    extractor = PlantLeafFeatureExtractor(image_size=128)
    rng = np.random.default_rng(42)
    synthetic_rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    vec = extractor.extract_from_rgb(synthetic_rgb)
    assert vec.shape == (extractor.num_features,)
    assert len(extractor.feature_names) == extractor.num_features
    assert np.all(np.isfinite(vec))


def test_healthy_green_vs_yellow_lesion_sensitivity() -> None:
    extractor = PlantLeafFeatureExtractor(image_size=128)
    green_leaf = np.zeros((128, 128, 3), dtype=np.uint8)
    green_leaf[:, :] = [30, 170, 35]

    yellow_leaf = np.zeros((128, 128, 3), dtype=np.uint8)
    yellow_leaf[:, :] = [190, 175, 25]

    feat_green = dict(zip(extractor.feature_names, extractor.extract_from_rgb(green_leaf)))
    feat_yellow = dict(zip(extractor.feature_names, extractor.extract_from_rgb(yellow_leaf)))

    assert feat_green["veg_exg_mean"] > feat_yellow["veg_exg_mean"]
    assert feat_yellow["veg_yellow_brown_ratio"] > feat_green["veg_yellow_brown_ratio"]
