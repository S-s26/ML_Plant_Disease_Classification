"""Publication-quality plotting helpers for EDA, feature importance, training curves, and model evaluation."""

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

sns.set_theme(style="whitegrid", context="notebook")


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot 38-class sample counts (colored by Healthy vs Diseased) and Binary pie/bar chart."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Binary Healthy vs Diseased & Crop-wise counts
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    bin_counts = df["binary_label"].value_counts()
    colors_bin = ["#2e8b57" if lbl == "healthy" else "#d95f02" for lbl in bin_counts.index]
    axes[0].bar(bin_counts.index, bin_counts.values, color=colors_bin, width=0.55)
    axes[0].set_title("Healthy vs. Diseased Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    for idx, val in enumerate(bin_counts.values):
        pct = 100.0 * val / len(df)
        axes[0].text(idx, val + 300, f"{val:,}\n({pct:.1f}%)", ha="center", fontweight="bold")

    crop_bin = df.groupby(["crop", "binary_label"]).size().unstack(fill_value=0)
    crop_bin = crop_bin.loc[crop_bin.sum(axis=1).sort_values(ascending=True).index]
    crop_bin.plot(
        kind="barh",
        stacked=True,
        ax=axes[1],
        color={"healthy": "#2e8b57", "diseased": "#d95f02"},
    )
    axes[1].set_title("Image Counts by Crop Species", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Images")
    axes[1].set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_dir / "eda_binary_and_crop_distribution.png", dpi=200)
    plt.close(fig)

    # 2. All 38 classes horizontal bar chart
    cls_counts = (
        df.groupby(["class_name", "binary_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(12, 13))
    bar_colors = [
        "#2e8b57" if b == "healthy" else "#3b6ea5" for b in cls_counts["binary_label"]
    ]
    ax.barh(cls_counts["class_name"], cls_counts["count"], color=bar_colors)
    ax.set_title("PlantVillage 38-Class Distribution (Green = Healthy, Blue = Diseased)", fontweight="bold")
    ax.set_xlabel("Number of Images")
    ax.tick_params(axis="y", labelsize=8)
    for idx, val in enumerate(cls_counts["count"]):
        ax.text(val + 30, idx, str(val), va="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(out_dir / "eda_38_class_distribution.png", dpi=200)
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, root_dir: Path, out_dir: Path, n_samples: int = 12) -> None:
    """Plot a balanced grid of sample healthy & diseased leaves."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    healthy_df = df[df["binary_label"] == "healthy"].sample(
        n=min(n_samples // 2, (df["binary_label"] == "healthy").sum()), random_state=42
    )
    diseased_df = df[df["binary_label"] == "diseased"].sample(
        n=min(n_samples // 2, (df["binary_label"] == "diseased").sum()), random_state=42
    )
    combined = pd.concat([healthy_df, diseased_df], ignore_index=True)

    cols = 4
    rows = int(np.ceil(len(combined) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.3 * rows))
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis("off")

    for idx, row in combined.iterrows():
        img = Image.open(root_dir / row["rel_path"]).convert("RGB")
        axes_flat[idx].imshow(img)
        color = "#1f7a1f" if row["binary_label"] == "healthy" else "#b22222"
        title = row["class_name"].replace("___", "\n")
        axes_flat[idx].set_title(title, fontsize=9, color=color, fontweight="bold")
    plt.suptitle("Sample PlantVillage Leaf Images (Healthy vs. Diseased)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "eda_sample_leaves.png", dpi=200)
    plt.close(fig)


def plot_training_history(history: Dict[str, List[float]], title: str, out_path: Path) -> None:
    """Plot train/val loss and accuracy learning curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], marker="o", label="Train Acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], marker="s", label="Val Acc")
    axes[1].set_title(f"{title} - Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_binary_confusion_and_roc(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    out_dir: Path,
    filename_prefix: str = "binary",
) -> None:
    """Plot ROC comparison across models and binary confusion matrices."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined ROC curve
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probs_pos in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate (Healthy flagged as Diseased)")
    ax.set_ylabel("True Positive Rate (Diseased Recall)")
    ax.set_title("Healthy vs. Diseased ROC Curve Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_roc_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Confusion matrices side-by-side
    n_models = len(model_probs)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]
    for ax, (name, probs_pos) in zip(axes, model_probs.items()):
        preds = (probs_pos >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Diseased"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(False)
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_multiclass_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    """Plot normalized 38x38 confusion matrix heatmap."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Row-Normalized Recall"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    title: str,
    out_path: Path,
    top_k: int = 20,
) -> None:
    """Plot top-k most important handcrafted image features."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([feature_names[i] for i in indices], importances[indices], color="#2a7b88")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Gini Feature Importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_model_benchmark_bar(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot clean horizontal bar chart comparing Accuracy and Macro-F1 across models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    melted = results_df.melt(
        id_vars=["model", "task"],
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    melted["label"] = melted["model"] + " (" + melted["task"] + ")"
    sns.barplot(data=melted, y="label", x="score", hue="metric", ax=ax, palette=["#3b6ea5", "#2e8b57"])
    ax.set_xlim(0.75, 1.06)
    ax.set_xlabel("Score (Test Split)")
    ax.set_ylabel("")
    ax.set_title("Model Performance Benchmark (Held-Out Test Split)", fontsize=13, fontweight="bold", pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=2, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8.5, padding=3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
