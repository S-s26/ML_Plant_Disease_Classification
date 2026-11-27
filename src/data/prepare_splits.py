"""Prepare deterministic leaf-grouped stratified train/val/test splits and integrity metadata."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.utils.io import ensure_dir, load_config, resolve_path, save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BINARY_CLASSES = ["healthy", "diseased"]


def parse_class_folder(folder_name: str) -> Tuple[str, str, int, str]:
    """Parse 'Crop___Condition' into (crop, condition, binary_id, binary_label)."""
    if "___" in folder_name:
        crop, condition = folder_name.split("___", maxsplit=1)
    else:
        crop, condition = folder_name, "unknown"
    is_healthy = condition.strip().lower() == "healthy"
    binary_id = 0 if is_healthy else 1
    binary_label = "healthy" if is_healthy else "diseased"
    return crop, condition, binary_id, binary_label


def resolve_leaf_id(filename: str, class_name: str, leaf_map: Dict[str, List[str]]) -> str:
    """Resolve physical leaf ID using PlantVillage leaf-map.json + official identifier rules."""
    ident = filename.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".webp"):
        ident = ident.replace(ext, "")
    lookup_key = ident.strip().lower()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion
    return f"{class_name}::fallback_{lookup_key}"


def _inspect_image(args: Tuple[Path, Path, str, int, Dict[str, List[str]]]) -> Dict:
    """Inspect single image header, size, MD5 hash, and physical leaf group."""
    img_path, repo_root, class_name, class_id, leaf_map = args
    crop, condition, binary_id, binary_label = parse_class_folder(class_name)
    rel_path = img_path.relative_to(repo_root).as_posix()
    leaf_id = resolve_leaf_id(img_path.name, class_name, leaf_map)

    width, height = -1, -1
    corrupt = False
    md5_hex = ""
    try:
        raw_bytes = img_path.read_bytes()
        md5_hex = hashlib.md5(raw_bytes).hexdigest()
        with Image.open(img_path) as im:
            width, height = im.size
    except Exception:
        corrupt = True

    return {
        "rel_path": rel_path,
        "filename": img_path.name,
        "leaf_id": leaf_id,
        "class_id": class_id,
        "class_name": class_name,
        "crop": crop,
        "condition": condition,
        "binary_id": binary_id,
        "binary_label": binary_label,
        "width": width,
        "height": height,
        "md5": md5_hex,
        "corrupt": corrupt,
    }


def prepare_splits(config_path: str | None = None, dedupe_exact: bool = True) -> pd.DataFrame:
    """Scan raw/color images, check integrity, deduplicate exact MD5s, and run leaf-grouped split."""
    cfg = load_config(config_path)
    color_dir = resolve_path(cfg["paths"]["raw_color_dir"])
    leaf_map_path = resolve_path(cfg["paths"]["leaf_grouping_dir"]) / "leaf-map.json"
    splits_csv = resolve_path(cfg["paths"]["splits_csv"])
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    ensure_dir(cfg["paths"]["processed_dir"])

    if not color_dir.exists():
        raise FileNotFoundError(
            f"Raw color dataset directory not found at {color_dir}. "
            "Run `python -m src.data.download_dataset` first."
        )

    leaf_map: Dict[str, List[str]] = {}
    if leaf_map_path.exists():
        with leaf_map_path.open("r", encoding="utf-8") as f:
            leaf_map = json.load(f)
        print(f"Loaded leaf-map.json with {len(leaf_map):,} leaf entries for leakage-free splitting.")

    class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    class_to_id = {d.name: idx for idx, d in enumerate(class_dirs)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}

    tasks: List[Tuple[Path, Path, str, int, Dict[str, List[str]]]] = []
    repo_root = resolve_path(".")
    for class_dir in class_dirs:
        cid = class_to_id[class_dir.name]
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                tasks.append((file_path, repo_root, class_dir.name, cid, leaf_map))

    print(f"Scanning {len(tasks)} images across {len(class_dirs)} classes...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(tqdm(pool.map(_inspect_image, tasks), total=len(tasks), desc="Scanning images"))

    df = pd.DataFrame(records)
    n_total = len(df)
    n_corrupt = int(df["corrupt"].sum())
    df = df[~df["corrupt"]].copy().reset_index(drop=True)

    n_exact_dups = int(df.duplicated(subset=["md5"]).sum())
    if dedupe_exact and n_exact_dups > 0:
        df = df.drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)

    # Leaf-grouped 7-fold split -> 5 folds (~71.4%) train, 1 fold (~14.3%) val, 1 fold (~14.3%) test
    seed = int(cfg["project"]["seed"])
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df["class_id"], groups=df["leaf_id"]))
    val_idx = folds[0][1]
    test_idx = folds[1][1]

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Sanity assertion: zero physical leaf overlap and all 38 classes present in every split
    train_leaves = set(df.loc[df["split"] == "train", "leaf_id"])
    val_leaves = set(df.loc[df["split"] == "val", "leaf_id"])
    test_leaves = set(df.loc[df["split"] == "test", "leaf_id"])
    assert not (train_leaves & val_leaves), "Leaf leakage detected between train and val!"
    assert not (train_leaves & test_leaves), "Leaf leakage detected between train and test!"
    assert not (val_leaves & test_leaves), "Leaf leakage detected between val and test!"
    for split in ("train", "val", "test"):
        n_cls_in_split = df.loc[df["split"] == split, "class_id"].nunique()
        assert n_cls_in_split == len(class_dirs), f"Split {split} missing classes ({n_cls_in_split})"

    df.to_csv(splits_csv, index=False)

    save_json(
        {
            "num_classes": len(class_dirs),
            "class_names": [id_to_class[i] for i in range(len(class_dirs))],
            "class_to_id": class_to_id,
            "binary_classes": BINARY_CLASSES,
            "binary_to_id": {"healthy": 0, "diseased": 1},
            "healthy_class_ids": sorted(
                df.loc[df["binary_id"] == 0, "class_id"].unique().tolist()
            ),
        },
        models_dir / "class_mapping.json",
    )

    dim_counts = df[["width", "height"]].value_counts().head(10).to_dict()
    summary = {
        "total_files_found": n_total,
        "corrupt_files_removed": n_corrupt,
        "exact_md5_duplicates_found": n_exact_dups,
        "deduplicated_exact_md5": dedupe_exact,
        "usable_images": int(len(df)),
        "unique_leaf_groups": int(df["leaf_id"].nunique()),
        "split_strategy": "StratifiedGroupKFold(n_splits=7, groups=leaf_id) -> 5:1:1 train:val:test",
        "num_classes": len(class_dirs),
        "num_crops": int(df["crop"].nunique()),
        "healthy_images": int((df["binary_id"] == 0).sum()),
        "diseased_images": int((df["binary_id"] == 1).sum()),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_leaf_group_counts": {
            s: int(df.loc[df["split"] == s, "leaf_id"].nunique()) for s in ("train", "val", "test")
        },
        "image_dimensions_unique": {f"{w}x{h}": int(cnt) for (w, h), cnt in dim_counts.items()},
    }
    save_json(summary, metrics_dir / "dataset_summary.json")
    print(f"Saved leaf-grouped dataset splits to {splits_csv}")
    print(f"Summary: {summary}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leaf-grouped stratified dataset splits.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact MD5 duplicate images instead of deduplicating before splitting",
    )
    args = parser.parse_args()
    prepare_splits(args.config, dedupe_exact=not args.keep_duplicates)


if __name__ == "__main__":
    main()
