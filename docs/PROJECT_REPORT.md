# Plant Disease Classification Using Image Features and Neural Networks — Project Report

**Course / Lab:** Machine Learning Lab Mini-Project (Problem Statement 5)  
**Dataset:** [PlantVillage Dataset (`spMohanty/PlantVillage-Dataset`)](https://github.com/spMohanty/PlantVillage-Dataset)  
**Environment:** Python 3.13, PyTorch 2.11.0+cu128, Scikit-Learn 1.6.1, OpenCV 5.0.0, Scikit-Image 0.25.0, Streamlit 1.45.1  

---

## 1. Abstract

Early and accurate identification of plant leaf diseases is essential for preventing crop yield loss and reducing unnecessary pesticide usage. This project implements a complete, leakage-free machine learning and deep learning system for **Plant Disease Classification** on the PlantVillage dataset (`54,284` deduplicated RGB images spanning `14` crops and `38` classes, grouped into `Healthy` vs. `Diseased`). To address both parts of the problem statement — *image features* and *neural networks* — we design and benchmark two complementary paradigms: (1) a **114-dimensional handcrafted computer vision feature pipeline** (RGB/HSV/CIELAB statistics and histograms, agricultural vegetation/lesion indices, GLCM texture, uniform LBP, Hu moments, and Sobel/Canny edges) paired with Logistic Regression, Random Forest, and a PyTorch **Feature-MLP Neural Network**; and (2) end-to-end spatial deep learning using a **4-stage Custom CNN built from scratch** and a **MobileNetV2 Transfer Learning CNN**, augmented with **Grad-CAM** visual explainability and an interactive **Streamlit** web application. Enforcing strict **physical leaf-grouped stratified splitting** (`StratifiedGroupKFold` over `20,015` leaf groups) guarantees zero leaf leakage across Train/Validation/Test splits. On the `7,825`-image unseen test set, the Feature-MLP achieves `98.94%` Binary (`0.9993` ROC-AUC) and `96.10%` 38-class accuracy, the Custom CNN achieves `97.65%` 38-class and `99.71%` rolled-up Binary accuracy, and MobileNetV2 achieves **`99.64%` 38-class accuracy (`99.49%` Macro-F1)** and **`99.95%` Binary Healthy-vs-Diseased accuracy (`0.9999` ROC-AUC)**.

---

## 2. Problem Statement & Objectives

**Lab Problem Statement:**  
> *5. Plant Disease Classification: Classify plant leaf images into healthy and diseased categories using image features and neural networks.*

**Project Objectives:**
1. Acquire and audit the official PlantVillage `raw/color` dataset and eliminate duplicate and leaf-level data leakage across splits.
2. Engineer an interpretable set of domain-specific **handcrafted image features** (color, chlorosis/lesion ratios, texture, and shape/edge descriptors) and evaluate classical ML vs. a Feature-MLP Neural Network.
3. Implement a **Custom CNN from scratch** and a **Transfer Learning CNN (MobileNetV2)**, comparing direct binary classification against 38-class fine-grained classification rolled up to Healthy vs. Diseased.
4. Provide model explainability via **Random Forest feature importance** and **Grad-CAM heatmaps**, packaged inside a tested codebase and interactive Streamlit diagnostic app.

---

## 3. Dataset & Exploratory Data Analysis

From `spMohanty/PlantVillage-Dataset` (`raw/color` + `leaf_grouping/leaf-map.json`):
- **Raw Scan:** `54,305` JPEG images (`256 x 256` RGB), `0` corrupt files, `21` exact MD5 duplicates removed -> **`54,284` usable images**.
- **Categories:** `38` crop-condition classes (`12` Healthy classes = `15,071` images / `27.76%`; `26` Diseased classes = `39,213` images / `72.24%`) across `14` crop species.
- **Leakage-Free Splitting:** Using `leaf-map.json` and `StratifiedGroupKFold(n_splits=7)` on `20,015` unique physical leaf groups:
  - **Train:** `38,741` images (`14,298` leaf groups)
  - **Validation:** `7,718` images (`2,858` leaf groups)
  - **Test:** `7,825` images (`2,859` leaf groups)
- **Confounding Discovery:** Five crops appear exclusively in one binary category (`Blueberry`, `Raspberry`, `Soybean` = healthy only; `Orange`, `Squash` = diseased only), motivating multi-class training + binary rollup.

*(See [DATASET_AND_EDA.md](DATASET_AND_EDA.md) and `outputs/figures/eda_*.png` for full tables and visualizations.)*

---

## 4. System Architecture

```text
PlantVillage RGB Leaf Image (256x256)
   │
   ├──► Pipeline A: Image Feature Engineering (128x128)
   │       ├── Color Spaces: RGB, HSV, CIELAB stats + histograms (66 dims)
   │       ├── Vegetation/Lesion Indices: ExG, GRVI, yellow/brown, necrotic (8 dims)
   │       ├── Texture: GLCM (d=1,3 across 4 angles) + Uniform LBP (30 dims)
   │       └── Shape & Structure: Hu Moments + Sobel + Canny density (10 dims)
   │             │
   │             ▼ StandardScaler (fit on Train only)
   │       ┌──────────────────────────┬──────────────────────────┐
   │       │ Classical ML Baselines   │ Feature-MLP (PyTorch)    │
   │       │ (LogReg, Random Forest)  │ 114 -> 256 -> 128 -> 64  │
   │       └──────────────────────────┴──────────────────────────┘
   │
   └──► Pipeline B: End-to-End Convolutional Neural Networks (224x224)
           ├── Custom CNN (From Scratch, 4 Conv Blocks + GAP, 1.21M params)
           └── Transfer CNN (MobileNetV2 IMAGENET1K_V2, 2.27M params)
                 │
                 ├──► 38-Class Prediction + Rolled-Up Healthy vs. Diseased
                 └──► Grad-CAM Visual Heatmap Overlay
```

---

## 5. Methodology & Implementation Summary

1. **Data Acquisition & Splitting (`src/data/`):** Sparse Git checkout downloads only `raw/color` and `leaf_grouping`, saving >1.3 GB. `prepare_splits.py` validates images, removes exact MD5 duplicates, resolves physical `leaf_id`s, and produces deterministic CSV splits + `models/class_mapping.json`.
2. **Feature Extraction (`src/features/`):** Multiprocess extraction builds `data/processed/handcrafted_features.npz` (`54,284 x 114`) in ~2.5 minutes.
3. **Training (`src/training/`):**
   - `train_classical_mlp.py` trains Logistic Regression, Random Forest, and PyTorch Feature-MLP for both Binary and 38-Class tasks.
   - `train_cnn.py` trains Custom CNN (Binary & 38-Class) and MobileNetV2 Transfer CNN (38-Class) on RTX 4050 GPU using `AMP (float16)`, `AdamW`, `CosineAnnealingLR`, square-root inverse class weights, and early stopping.
4. **Evaluation & App (`src/training/evaluate_all.py`, `src/inference/predictor.py`, `app/app.py`):** Evaluates all models uniformly on the held-out `7,825`-image test set, computes rolled-up Binary ROC-AUC & confusion matrices, generates Grad-CAM overlays, and serves predictions via a 3-tab Streamlit UI.

---

## 6. Experimental Results & Discussion

| Model | Task | Accuracy | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression (Features) | Binary | 93.10% | 91.56% | 0.9833 |
| Random Forest (Features) | Binary | 97.78% | 97.14% | 0.9973 |
| Feature-MLP (Features) | Binary | 98.94% | 98.65% | 0.9993 |
| Custom CNN (Direct Binary) | Binary | 99.49% | 99.35% | 0.9998 |
| Custom CNN [38-Class Rollup] | Binary | 99.71% | 99.62% | 0.99996 |
| **Transfer CNN (MobileNetV2) [Rollup]** | **Binary** | **99.95%** | **99.93%** | **0.99986** |
| Random Forest (Features) | 38-Class | 91.55% | 87.53% | — |
| Logistic Regression (Features) | 38-Class | 92.61% | 90.49% | — |
| Feature-MLP (Features) | 38-Class | 96.10% | 94.43% | — |
| Custom CNN (From Scratch) | 38-Class | 97.65% | 96.15% | — |
| **Transfer CNN (MobileNetV2)** | **38-Class** | **99.64%** | **99.49%** | — |

**Key Takeaways:**
- **Handcrafted Features vs. Neural Networks:** Handcrafted features + Feature-MLP reach `98.94%` Binary and `96.10%` 38-class accuracy, outperforming classical Random Forest (`91.55%` 38-class) and Logistic Regression (`92.61%` 38-class), while CNNs push 38-class accuracy to `97.65%` (from scratch) and `99.64%` (MobileNetV2).
- **Explainability:** Random Forest confirms `veg_yellow_brown_ratio` as the #1 feature, and Grad-CAM verifies that MobileNetV2 focuses directly on fungal pustules and necrotic lesions rather than background margins.
- **Error Profile:** Only `28` errors out of `7,825` test samples in 38-class mode, mostly between closely related maize blights (`Gray_leaf_spot` vs. `Northern_Leaf_Blight`, `6` confusions) and tomato blights (`Early_blight` vs. `Septoria_leaf_spot`, `3` confusions).

---

## 7. Conclusion & Future Scope

We delivered an end-to-end, reproducible, well-tested plant pathology classification system fulfilling every requirement of Lab Problem 5. Future improvements include training on in-the-wild field datasets (e.g., PlantDoc), applying background segmentation / CutMix to eliminate controlled-lab background bias, adding out-of-distribution rejection for non-leaf uploads, and exporting quantized ONNX/TFLite models for offline smartphone deployment.

---

## 8. References

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7:1419.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
3. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
4. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *Textural Features for Image Classification* (GLCM). IEEE Trans. Systems, Man, and Cybernetics.
5. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*. IEEE TPAMI.
