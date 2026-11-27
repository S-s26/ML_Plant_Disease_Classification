"""Unified evaluation, EDA visualization, rolled-up binary analysis, and Grad-CAM showcase."""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import transforms

from src.data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.custom_cnn import PlantDiseaseCNN
from src.models.gradcam import GradCAM, overlay_gradcam
from src.models.transfer_cnn import TransferPlantCNN
from src.utils.io import ensure_dir, load_config, load_json, resolve_path, save_json
from src.utils.visualization import (
    plot_binary_confusion_and_roc,
    plot_class_distribution,
    plot_model_benchmark_bar,
    plot_multiclass_confusion,
    plot_sample_grid,
)


def _eval_row(
    model_name: str,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_probs: np.ndarray | None = None,
) -> Dict:
    row = {
        "model": model_name,
        "task": task,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pos_probs)) if pos_probs is not None else np.nan,
    }
    return row


def generate_gradcam_showcase(
    df_test: pd.DataFrame,
    ckpt_path: Path,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Create a 6-sample Grad-CAM visualization grid on test leaf images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    backbone = ckpt.get("backbone_name", "mobilenet_v2")
    image_size = int(ckpt.get("image_size", 224))

    if ckpt.get("arch") == "custom":
        model = PlantDiseaseCNN(num_classes=int(ckpt["num_classes"]))
    else:
        model = TransferPlantCNN(
            num_classes=int(ckpt["num_classes"]), backbone_name=backbone, pretrained=False
        )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    healthy_samples = df_test[df_test["binary_label"] == "healthy"].sample(
        n=min(2, (df_test["binary_label"] == "healthy").sum()), random_state=42
    )
    diseased_samples = df_test[df_test["binary_label"] == "diseased"].sample(
        n=min(4, (df_test["binary_label"] == "diseased").sum()), random_state=42
    )
    chosen = pd.concat([healthy_samples, diseased_samples], ignore_index=True)

    root = resolve_path(".")
    gradcam = GradCAM(model, model.get_target_conv_layer())

    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 3.2 * len(chosen)))
    try:
        for idx, row in chosen.iterrows():
            with Image.open(root / row["rel_path"]) as im:
                rgb_orig = np.array(im.convert("RGB").resize((image_size, image_size)))
                inp = tf(im.convert("RGB")).unsqueeze(0).to(device)

            heatmap, pred_idx, conf = gradcam.generate(inp)
            overlay = overlay_gradcam(rgb_orig, heatmap, alpha=0.45)

            true_name = row["class_name"].replace("___", ": ")
            pred_name = class_names[pred_idx].replace("___", ": ")
            ok = pred_idx == int(row["class_id"])

            axes[idx, 0].imshow(rgb_orig)
            axes[idx, 0].set_title(f"True: {true_name}", fontsize=9, fontweight="bold")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(overlay)
            axes[idx, 1].set_title(
                f"Pred: {pred_name}\nConf: {conf*100:.1f}% ({'Correct' if ok else 'Error'})",
                fontsize=9,
                color="#1f7a1f" if ok else "#b22222",
                fontweight="bold",
            )
            axes[idx, 1].axis("off")
    finally:
        gradcam.remove_hooks()

    plt.suptitle("Grad-CAM Visual Explainability (Transfer CNN on Test Samples)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_all(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    splits_csv = resolve_path(cfg["paths"]["splits_csv"])
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])
    class_map = load_json(models_dir / "class_mapping.json")

    df = pd.read_csv(splits_csv)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    root = resolve_path(".")

    # 1. EDA Figures
    print("Generating EDA visualizations...")
    plot_class_distribution(df, figures_dir)
    plot_sample_grid(df, root, figures_dir, n_samples=12)

    class_names: List[str] = class_map["class_names"]
    healthy_ids: List[int] = class_map["healthy_class_ids"]
    y_bin_test = df_test["binary_id"].to_numpy(dtype=np.int64)
    y_cls_test = df_test["class_id"].to_numpy(dtype=np.int64)

    rows: List[Dict] = []
    binary_prob_dict: Dict[str, np.ndarray] = {}

    # 2. Classical ML & Feature-MLP predictions
    feat_npz_path = metrics_dir / "feature_test_predictions.npz"
    if feat_npz_path.exists():
        feat_preds = np.load(feat_npz_path)
        for key, label in [
            ("logreg", "Logistic Regression (Features)"),
            ("rf", "Random Forest (Features)"),
            ("mlp", "Feature-MLP (Features)"),
        ]:
            bin_probs = feat_preds[f"probs_{key}_binary"]
            rows.append(
                _eval_row(
                    label,
                    "Binary",
                    feat_preds["y_bin_test"],
                    bin_probs.argmax(axis=1),
                    pos_probs=bin_probs[:, 1],
                )
            )
            binary_prob_dict[label] = bin_probs[:, 1]

            cls_probs = feat_preds[f"probs_{key}_multiclass"]
            rows.append(
                _eval_row(
                    label,
                    "38-Class",
                    feat_preds["y_cls_test"],
                    cls_probs.argmax(axis=1),
                )
            )

    # 3. Direct Binary Custom CNN (if trained)
    bin_cnn_npz = metrics_dir / "custom_cnn_binary_test_predictions.npz"
    if bin_cnn_npz.exists():
        d = np.load(bin_cnn_npz)
        probs = d["probs"]
        rows.append(
            _eval_row(
                "Custom CNN (Direct Binary)",
                "Binary",
                d["y_true"],
                probs.argmax(axis=1),
                pos_probs=probs[:, 1],
            )
        )
        binary_prob_dict["Custom CNN (Direct)"] = probs[:, 1]

    # 4. Multiclass CNNs + Rolled-Up Binary Evaluation
    backbone = str(cfg["training"]["transfer_backbone"])
    cnn_specs = [
        ("custom_cnn_multiclass", "Custom CNN (From Scratch)", "multiclass_confusion_custom_cnn.png"),
        (f"transfer_{backbone}_multiclass", f"Transfer CNN ({backbone})", "multiclass_confusion_transfer_cnn.png"),
    ]
    for run_name, display_name, cm_filename in cnn_specs:
        npz_path = metrics_dir / f"{run_name}_test_predictions.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        probs_38 = d["probs"]
        y_true_38 = d["y_true"]
        preds_38 = probs_38.argmax(axis=1)
        rows.append(_eval_row(display_name, "38-Class", y_true_38, preds_38))

        # Roll up 38-class probability into Binary (0=healthy, 1=diseased)
        p_healthy = probs_38[:, healthy_ids].sum(axis=1)
        p_diseased = np.clip(1.0 - p_healthy, 0.0, 1.0)
        preds_bin_rollup = (p_diseased >= 0.5).astype(int)
        rollup_name = f"{display_name} [Rollup]"
        rows.append(
            _eval_row(
                rollup_name,
                "Binary",
                y_bin_test,
                preds_bin_rollup,
                pos_probs=p_diseased,
            )
        )
        binary_prob_dict[rollup_name] = p_diseased

        plot_multiclass_confusion(
            y_true=y_true_38,
            y_pred=preds_38,
            class_names=class_names,
            title=f"Normalized 38-Class Confusion Matrix - {display_name}",
            out_path=figures_dir / cm_filename,
        )

        if "transfer" in run_name:
            rep = classification_report(
                y_true_38,
                preds_38,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            rep_df = pd.DataFrame(rep).transpose()
            rep_df.to_csv(metrics_dir / "transfer_cnn_38class_report.csv")

            # Top misclassified pairs for error analysis
            mis_mask = preds_38 != y_true_38
            err_df = pd.DataFrame(
                {
                    "true_class": [class_names[i] for i in y_true_38[mis_mask]],
                    "pred_class": [class_names[i] for i in preds_38[mis_mask]],
                }
            )
            top_errors = (
                err_df.groupby(["true_class", "pred_class"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(15)
                .to_dict(orient="records")
            )
            save_json(
                {
                    "total_test_samples": int(len(y_true_38)),
                    "total_misclassified": int(mis_mask.sum()),
                    "top_confused_pairs": top_errors,
                },
                metrics_dir / "error_analysis_transfer_cnn.json",
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(metrics_dir / "benchmark_summary.csv", index=False)
    save_json(results_df.to_dict(orient="records"), metrics_dir / "benchmark_summary.json")

    if binary_prob_dict:
        plot_binary_confusion_and_roc(y_bin_test, binary_prob_dict, figures_dir)
    if not results_df.empty:
        plot_model_benchmark_bar(results_df, figures_dir / "model_benchmark_comparison.png")

    transfer_ckpt = models_dir / f"transfer_{backbone}_multiclass.pt"
    if transfer_ckpt.exists():
        print("Generating Grad-CAM showcase...")
        generate_gradcam_showcase(
            df_test=df_test,
            ckpt_path=transfer_ckpt,
            class_names=class_names,
            out_path=figures_dir / "gradcam_showcase.png",
        )

    print("\n=== Final Benchmark Summary ===")
    print(results_df.to_string(index=False))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all trained models and generate figures.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate_all(args.config)


if __name__ == "__main__":
    main()
