"""Train classical ML baselines (LogReg, RandomForest) and Feature-MLP Neural Network on handcrafted features."""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.classical_and_mlp import FeatureMLP
from src.utils.io import ensure_dir, load_config, load_json, resolve_path, save_json
from src.utils.seed import set_seed
from src.utils.visualization import plot_feature_importance, plot_training_history


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, is_binary: bool
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if is_binary:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
    return metrics


def _sqrt_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** 0.5
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_feature_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    ckpt_path: Path,
) -> Tuple[np.ndarray, Dict[str, list[float]]]:
    """Train PyTorch FeatureMLP with early stopping and return test probabilities + history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = FeatureMLP(in_features=X_train.shape[1], num_classes=num_classes).to(device)
    weights = _sqrt_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * bx.size(0)
            tr_correct += int((logits.argmax(dim=1) == by).sum().item())
            tr_total += bx.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += float(loss.item()) * bx.size(0)
                val_correct += int((logits.argmax(dim=1) == by).sum().item())
                val_total += bx.size(0)

        ep_tr_loss = tr_loss / tr_total
        ep_val_loss = val_loss / val_total
        ep_tr_acc = tr_correct / tr_total
        ep_val_acc = val_correct / val_total
        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_acc"].append(ep_val_acc)

        if ep_val_loss < best_val_loss - 1e-4:
            best_val_loss = ep_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"FeatureMLP early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "in_features": int(X_train.shape[1]),
            "num_classes": int(num_classes),
        },
        ckpt_path,
    )
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(device)
        probs = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
    return probs, history


def run_classical_and_mlp(config_path: str | None = None) -> Dict:
    cfg = load_config(config_path)
    set_seed(int(cfg["project"]["seed"]))

    features_npz = resolve_path(cfg["paths"]["features_npz"])
    meta = load_json(cfg["paths"]["features_meta_json"])
    class_map = load_json(resolve_path(cfg["paths"]["models_dir"]) / "class_mapping.json")
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])

    data = np.load(features_npz)
    X_train_raw, X_val_raw, X_test_raw = data["X_train"], data["X_val"], data["X_test"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")

    results: Dict[str, Dict] = {}
    pred_arrays: Dict[str, np.ndarray] = {
        "y_bin_test": data["y_bin_test"],
        "y_cls_test": data["y_cls_test"],
    }

    feature_names = meta["feature_names"]
    seed = int(cfg["project"]["seed"])
    num_multiclass = int(class_map["num_classes"])

    for task, y_key, n_classes in [
        ("binary", "bin", 2),
        ("multiclass", "cls", num_multiclass),
    ]:
        y_train = data[f"y_{y_key}_train"]
        y_val = data[f"y_{y_key}_val"]
        y_test = data[f"y_{y_key}_test"]
        is_bin = task == "binary"

        print(f"\n=== Training Handcrafted-Feature Models ({task}, {n_classes} classes) ===")

        # 1. Logistic Regression
        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=2.0,
            random_state=seed,
        )
        logreg.fit(X_train, y_train)
        probs_lr = logreg.predict_proba(X_test)
        preds_lr = probs_lr.argmax(axis=1)
        results[f"logreg_{task}"] = compute_metrics(y_test, preds_lr, probs_lr, is_bin)
        pred_arrays[f"probs_logreg_{task}"] = probs_lr
        joblib.dump(logreg, models_dir / f"logreg_{task}.joblib")
        print(f"LogisticRegression ({task}): {results[f'logreg_{task}']}")

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(X_train, y_train)
        probs_rf = rf.predict_proba(X_test)
        preds_rf = probs_rf.argmax(axis=1)
        results[f"random_forest_{task}"] = compute_metrics(y_test, preds_rf, probs_rf, is_bin)
        pred_arrays[f"probs_rf_{task}"] = probs_rf
        joblib.dump(rf, models_dir / f"random_forest_{task}.joblib")
        print(f"RandomForest ({task}): {results[f'random_forest_{task}']}")

        plot_feature_importance(
            feature_names=feature_names,
            importances=rf.feature_importances_,
            title=f"Top 20 Handcrafted Image Features ({task.title()} - Random Forest)",
            out_path=figures_dir / f"feature_importance_{task}.png",
        )

        # 3. Feature-MLP Neural Network
        probs_mlp, hist_mlp = train_feature_mlp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            num_classes=n_classes,
            epochs=int(cfg["training"]["feature_mlp_epochs"]),
            lr=float(cfg["training"]["learning_rate_mlp"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            patience=6,
            ckpt_path=models_dir / f"feature_mlp_{task}.pt",
        )
        preds_mlp = probs_mlp.argmax(axis=1)
        results[f"feature_mlp_{task}"] = compute_metrics(y_test, preds_mlp, probs_mlp, is_bin)
        pred_arrays[f"probs_mlp_{task}"] = probs_mlp
        print(f"Feature-MLP ({task}): {results[f'feature_mlp_{task}']}")

        plot_training_history(
            hist_mlp,
            title=f"Feature-MLP ({task.title()})",
            out_path=figures_dir / f"history_feature_mlp_{task}.png",
        )

    save_json(results, metrics_dir / "classical_mlp_results.json")
    np.savez_compressed(metrics_dir / "feature_test_predictions.npz", **pred_arrays)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML and Feature-MLP models.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_classical_and_mlp(args.config)


if __name__ == "__main__":
    main()
