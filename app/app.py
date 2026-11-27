"""Streamlit interactive lab demo for Plant Disease Classification."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import PlantDiseasePredictor
from src.utils.io import load_json, resolve_path

st.set_page_config(
    page_title="Plant Disease Classification | ML Lab Mini-Project",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource
def get_predictor() -> PlantDiseasePredictor:
    return PlantDiseasePredictor()


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not generated yet: `{path.name}`. Run evaluation first.")


def main() -> None:
    st.title("🌿 Plant Disease Classification")
    st.caption(
        "Classifying plant leaf images into Healthy vs. Diseased categories (and 38 PlantVillage classes) "
        "using Handcrafted Image Features + Neural Networks (Feature-MLP, Custom CNN, MobileNetV2 Transfer CNN)."
    )

    try:
        predictor = get_predictor()
        model_choices = predictor.available_models()
    except Exception as exc:
        st.error(
            f"Trained models could not be loaded yet ({exc}). "
            "Run `python scripts/run_pipeline.py` first."
        )
        return

    if not model_choices:
        st.warning("No trained checkpoints found in `models/`. Run the training pipeline first.")
        return

    tab_live, tab_bench, tab_eda = st.tabs(
        ["🔍 Live Leaf Diagnosis", "📊 Model Benchmark & Results", "🧬 Dataset EDA & Grad-CAM"]
    )

    with tab_live:
        col_left, col_right = st.columns([1, 1.3], gap="large")
        with col_left:
            selected_model = st.selectbox("Select Model Architecture", model_choices, index=0)
            source_mode = st.radio(
                "Input Source",
                ["Upload Leaf Image", "Pick Sample from Held-Out Test Split"],
                horizontal=True,
            )

            pil_img: Image.Image | None = None
            true_label_info: str | None = None

            if source_mode == "Upload Leaf Image":
                uploaded = st.file_uploader("Upload a plant leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
                if uploaded is not None:
                    pil_img = Image.open(uploaded).convert("RGB")
            else:
                splits_csv = resolve_path("data/processed/dataset_splits.csv")
                if splits_csv.exists():
                    df_test = pd.read_csv(splits_csv)
                    df_test = df_test[df_test["split"] == "test"].reset_index(drop=True)
                    classes = sorted(df_test["class_name"].unique().tolist())
                    chosen_cls = st.selectbox("Choose Ground-Truth Class", classes)
                    sub = df_test[df_test["class_name"] == chosen_cls].reset_index(drop=True)
                    sample_idx = st.slider("Sample Index", 0, max(0, len(sub) - 1), 0)
                    row = sub.iloc[sample_idx]
                    pil_img = Image.open(PROJECT_ROOT / row["rel_path"]).convert("RGB")
                    true_label_info = f"{row['class_name']} ({row['binary_label'].upper()})"
                else:
                    st.info("`dataset_splits.csv` not found. Upload an image instead.")

            if pil_img is not None:
                st.image(pil_img, caption="Input Leaf Image", use_container_width=True)
                if true_label_info:
                    st.info(f"**Ground Truth:** `{true_label_info}`")

        with col_right:
            if pil_img is not None:
                result = predictor.predict(pil_img, model_choice=selected_model, top_k=5)
                verdict = result["binary_verdict"]
                if verdict == "Healthy":
                    st.success(
                        f"### Status: HEALTHY ({result['healthy_probability']*100:.2f}% confidence)"
                    )
                else:
                    st.error(
                        f"### Status: DISEASED ({result['diseased_probability']*100:.2f}% confidence)"
                    )

                c1, c2 = st.columns(2)
                c1.metric("Top Predicted Class", result["top_class"].replace("___", " — "))
                c2.metric("Class Confidence", f"{result['top_confidence']*100:.2f}%")

                st.markdown("#### Top-5 Class Probabilities")
                top_df = pd.DataFrame(result["top_predictions"])
                top_df["class_name"] = top_df["class_name"].str.replace("___", " — ", regex=False)
                top_df["probability"] = (top_df["probability"] * 100).round(2)
                top_df.columns = ["Class ID", "Crop & Condition", "Confidence (%)"]
                st.dataframe(top_df[["Crop & Condition", "Confidence (%)"]], hide_index=True, use_container_width=True)

                if result["gradcam_overlay"] is not None:
                    st.markdown("#### Grad-CAM Visual Explanation")
                    st.image(
                        result["gradcam_overlay"],
                        caption="Regions most influencing the CNN prediction (Red = High Attention)",
                        width=340,
                    )

                st.markdown("#### Extracted Handcrafted Image Feature Indicators")
                feat_df = pd.DataFrame(
                    list(result["key_image_features"].items()),
                    columns=["Image Feature", "Measured Value"],
                )
                st.dataframe(feat_df, hide_index=True, use_container_width=True)
            else:
                st.info("Upload or select a leaf image on the left to run diagnosis.")

    with tab_bench:
        bench_csv = resolve_path("outputs/metrics/benchmark_summary.csv")
        figures_dir = resolve_path("outputs/figures")
        if bench_csv.exists():
            st.subheader("Held-Out Test Split Performance Across All Pipelines")
            bench_df = pd.read_csv(bench_csv)
            st.dataframe(bench_df, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            show_image_if_exists(figures_dir / "model_benchmark_comparison.png", "Overall Model Benchmark")
            show_image_if_exists(figures_dir / "binary_confusion_matrices.png", "Binary Confusion Matrices")
        with c2:
            show_image_if_exists(figures_dir / "binary_roc_comparison.png", "Healthy vs. Diseased ROC Curves")
            show_image_if_exists(figures_dir / "feature_importance_binary.png", "Top Handcrafted Image Features")

        show_image_if_exists(
            figures_dir / "multiclass_confusion_transfer_cnn.png",
            "38-Class Normalized Confusion Matrix (Transfer CNN)",
        )

    with tab_eda:
        summary_json = resolve_path("outputs/metrics/dataset_summary.json")
        figures_dir = resolve_path("outputs/figures")
        if summary_json.exists():
            summary = load_json(summary_json)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Usable Images", f"{summary['usable_images']:,}")
            m2.metric("Classes", summary["num_classes"])
            m3.metric("Healthy Images", f"{summary['healthy_images']:,}")
            m4.metric("Diseased Images", f"{summary['diseased_images']:,}")

        c1, c2 = st.columns(2)
        with c1:
            show_image_if_exists(figures_dir / "eda_binary_and_crop_distribution.png", "Crop & Binary Distribution")
            show_image_if_exists(figures_dir / "eda_sample_leaves.png", "Sample Leaf Images")
        with c2:
            show_image_if_exists(figures_dir / "eda_38_class_distribution.png", "38-Class Distribution")
            show_image_if_exists(figures_dir / "gradcam_showcase.png", "Grad-CAM Test Showcase")


if __name__ == "__main__":
    main()
