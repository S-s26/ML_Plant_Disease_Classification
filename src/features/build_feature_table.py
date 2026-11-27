"""Parallel handcrafted feature extraction across train/val/test dataset splits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extractor import PlantLeafFeatureExtractor
from src.utils.io import ensure_dir, load_config, resolve_path, save_json

_EXTRACTOR: PlantLeafFeatureExtractor | None = None


def _init_worker(image_size: int, lbp_radius: int, lbp_points: int) -> None:
    global _EXTRACTOR
    _EXTRACTOR = PlantLeafFeatureExtractor(
        image_size=image_size,
        lbp_radius=lbp_radius,
        lbp_points=lbp_points,
    )


def _extract_single(path_str: str) -> np.ndarray:
    assert _EXTRACTOR is not None
    return _EXTRACTOR.extract_from_path(path_str)


def build_feature_table(config_path: str | None = None, force: bool = False) -> Path:
    """Extract 114-dim handcrafted features for all images and save split arrays as .npz."""
    cfg = load_config(config_path)
    splits_csv = resolve_path(cfg["paths"]["splits_csv"])
    features_npz = resolve_path(cfg["paths"]["features_npz"])
    meta_json = resolve_path(cfg["paths"]["features_meta_json"])
    ensure_dir(cfg["paths"]["processed_dir"])

    if features_npz.exists() and not force:
        print(f"Cached feature table already exists at {features_npz}")
        return features_npz

    df = pd.read_csv(splits_csv)
    root = resolve_path(".")
    abs_paths = [str(root / rel) for rel in df["rel_path"].tolist()]

    feat_img_size = int(cfg["dataset"]["feature_image_size"])
    lbp_radius = int(cfg["features"]["lbp_radius"])
    lbp_points = int(cfg["features"]["lbp_points"])
    max_workers = int(cfg["features"]["max_workers"])

    extractor = PlantLeafFeatureExtractor(
        image_size=feat_img_size,
        lbp_radius=lbp_radius,
        lbp_points=lbp_points,
    )

    print(
        f"Extracting {extractor.num_features} handcrafted image features for {len(abs_paths)} images "
        f"(size={feat_img_size}x{feat_img_size}, workers={max_workers})..."
    )
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(feat_img_size, lbp_radius, lbp_points),
    ) as pool:
        rows = list(
            tqdm(pool.map(_extract_single, abs_paths, chunksize=64), total=len(abs_paths), desc="Features")
        )

    X = np.vstack(rows).astype(np.float32)
    y_cls = df["class_id"].to_numpy(dtype=np.int64)
    y_bin = df["binary_id"].to_numpy(dtype=np.int64)
    splits = df["split"].to_numpy()

    arrays = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        arrays[f"X_{split}"] = X[mask]
        arrays[f"y_cls_{split}"] = y_cls[mask]
        arrays[f"y_bin_{split}"] = y_bin[mask]

    np.savez_compressed(features_npz, **arrays)
    save_json(
        {
            "num_features": extractor.num_features,
            "feature_names": extractor.feature_names,
            "feature_image_size": feat_img_size,
            "counts": {
                split: int((splits == split).sum()) for split in ("train", "val", "test")
            },
        },
        meta_json,
    )
    print(f"Saved feature matrix {X.shape} to {features_npz}")
    return features_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Build handcrafted image feature dataset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()
    build_feature_table(args.config, force=args.force)


if __name__ == "__main__":
    main()
