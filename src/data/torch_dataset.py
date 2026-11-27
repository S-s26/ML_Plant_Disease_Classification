"""PyTorch Dataset, augmentations, and DataLoader factory for PlantVillage."""

from pathlib import Path
from typing import Dict, Literal, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from src.utils.io import resolve_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Build conservative agriculture-aware train and deterministic eval v2 transforms."""
    train_tf = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            # Mild color jitter so green-vs-yellow disease cues are preserved
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


class PlantVillageImageDataset(Dataset):
    """Map-style dataset backed by dataset_splits.csv."""

    def __init__(
        self,
        df: pd.DataFrame,
        task: Literal["multiclass", "binary"] = "multiclass",
        transform: transforms.Compose | None = None,
        root_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.task = task
        self.transform = transform
        self.root_dir = root_dir or resolve_path(".")
        self.target_col = "class_id" if task == "multiclass" else "binary_id"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["rel_path"]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.ToTensor()(rgb)
        label = int(row[self.target_col])
        return tensor, label


def create_dataloaders(
    splits_csv: Path,
    task: Literal["multiclass", "binary"] = "multiclass",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[Dict[str, DataLoader], Dict[str, pd.DataFrame]]:
    """Create train, val, and test DataLoaders along with split DataFrames."""
    df = pd.read_csv(splits_csv)
    tfs = build_transforms(image_size=image_size)
    pin_mem = torch.cuda.is_available()

    split_dfs: Dict[str, pd.DataFrame] = {}
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        sub_df = df[df["split"] == split].reset_index(drop=True)
        split_dfs[split] = sub_df
        ds = PlantVillageImageDataset(sub_df, task=task, transform=tfs[split])
        w = num_workers if split == "train" else min(2, num_workers)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=w,
            pin_memory=pin_mem,
            persistent_workers=(w > 0 and split == "train"),
            prefetch_factor=2 if w > 0 else None,
        )
    return loaders, split_dfs
