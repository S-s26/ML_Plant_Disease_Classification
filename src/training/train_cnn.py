"""Unified PyTorch CNN trainer for Custom CNN (from scratch) and Transfer Learning CNN."""

import argparse
import time
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from src.data.torch_dataset import create_dataloaders
from src.models.custom_cnn import PlantDiseaseCNN
from src.models.transfer_cnn import TransferPlantCNN
from src.utils.io import ensure_dir, load_config, load_json, resolve_path, save_json
from src.utils.seed import set_seed
from src.utils.visualization import plot_training_history


def _build_class_weights(df_train: pd.DataFrame, target_col: str, num_classes: int) -> torch.Tensor:
    counts = np.bincount(df_train[target_col].to_numpy(dtype=np.int64), minlength=num_classes).astype(
        np.float32
    )
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** 0.5
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _run_eval(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += float(loss.item()) * images.size(0)
            probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(targets_np)
    return avg_loss, targets_np, probs_np


def train_cnn(
    arch: Literal["custom", "transfer"],
    task: Literal["multiclass", "binary"] = "multiclass",
    config_path: str | None = None,
) -> Dict:
    """Train Custom CNN or Transfer CNN with AMP, CosineAnnealingLR, and early stopping."""
    cfg = load_config(config_path)
    set_seed(int(cfg["project"]["seed"]))

    splits_csv = resolve_path(cfg["paths"]["splits_csv"])
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])
    class_map = load_json(models_dir / "class_mapping.json")

    num_classes = 2 if task == "binary" else int(class_map["num_classes"])
    target_col = "binary_id" if task == "binary" else "class_id"

    loaders, split_dfs = create_dataloaders(
        splits_csv=splits_csv,
        task=task,
        image_size=int(cfg["dataset"]["image_size"]),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    backbone_name = str(cfg["training"]["transfer_backbone"])
    if arch == "custom":
        model: nn.Module = PlantDiseaseCNN(num_classes=num_classes)
        epochs = int(cfg["training"]["custom_cnn_epochs"])
        lr = float(cfg["training"]["learning_rate_custom"])
        run_name = f"custom_cnn_{task}"
    else:
        model = TransferPlantCNN(
            num_classes=num_classes, backbone_name=backbone_name, pretrained=True
        )
        epochs = int(cfg["training"]["transfer_cnn_epochs"])
        lr = float(cfg["training"]["learning_rate_transfer"])
        run_name = f"transfer_{backbone_name}_{task}"

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training {run_name} on {device} (params={n_params:,}, AMP={use_amp}) ===")

    weights = _build_class_weights(split_dfs["train"], target_col, num_classes).to(device)
    smoothing = 0.05 if task == "multiclass" else 0.0
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(cfg["training"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    patience = int(cfg["training"]["early_stopping_patience"])
    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    ckpt_path = models_dir / f"{run_name}.pt"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        pbar = tqdm(loaders["train"], desc=f"{run_name} Ep {epoch}/{epochs}", leave=False, mininterval=10.0)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            tr_loss += float(loss.item()) * images.size(0)
            tr_correct += int((logits.argmax(dim=1) == targets).sum().item())
            tr_total += images.size(0)
        scheduler.step()

        ep_tr_loss = tr_loss / tr_total
        ep_tr_acc = tr_correct / tr_total
        ep_val_loss, val_targets, val_probs = _run_eval(
            model, loaders["val"], criterion, device, use_amp
        )
        ep_val_acc = float((val_probs.argmax(axis=1) == val_targets).mean())

        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_acc"].append(ep_val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={ep_tr_loss:.4f} train_acc={ep_tr_acc*100:.2f}% | "
            f"val_loss={ep_val_loss:.4f} val_acc={ep_val_acc*100:.2f}%"
        )

        if ep_val_acc > best_val_acc + 1e-4 or (
            abs(ep_val_acc - best_val_acc) <= 1e-4 and ep_val_loss < best_val_loss
        ):
            best_val_acc = ep_val_acc
            best_val_loss = ep_val_loss
            no_improve = 0
            torch.save(
                {
                    "arch": arch,
                    "backbone_name": backbone_name if arch == "transfer" else "custom_cnn",
                    "task": task,
                    "num_classes": num_classes,
                    "image_size": int(cfg["dataset"]["image_size"]),
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping {run_name} at epoch {epoch}.")
                break

    elapsed_sec = time.time() - start_time
    plot_training_history(
        history,
        title=run_name.replace("_", " ").title(),
        out_path=figures_dir / f"history_{run_name}.png",
    )

    # Evaluate best checkpoint on Test split
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_targets, test_probs = _run_eval(
        model, loaders["test"], criterion, device, use_amp
    )
    test_preds = test_probs.argmax(axis=1)

    metrics = {
        "model": run_name,
        "arch": arch,
        "task": task,
        "num_params": int(n_params),
        "epochs_trained": len(history["train_loss"]),
        "train_seconds": round(elapsed_sec, 1),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(test_targets, test_preds)),
        "macro_precision": float(
            precision_score(test_targets, test_preds, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(test_targets, test_preds, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(test_targets, test_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_targets, test_preds, average="weighted", zero_division=0)),
    }
    if task == "binary":
        metrics["roc_auc"] = float(roc_auc_score(test_targets, test_probs[:, 1]))

    save_json(metrics, metrics_dir / f"{run_name}_metrics.json")
    np.savez_compressed(
        metrics_dir / f"{run_name}_test_predictions.npz",
        y_true=test_targets,
        probs=test_probs,
    )
    print(f"Test metrics for {run_name}: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Custom or Transfer CNN.")
    parser.add_argument("--arch", choices=["custom", "transfer"], required=True)
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train_cnn(arch=args.arch, task=args.task, config_path=args.config)


if __name__ == "__main__":
    main()
