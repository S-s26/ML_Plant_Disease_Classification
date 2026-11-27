# Architecture & Experiments

This document details the feature engineering pipeline, neural network architectures, training configurations, and comparative experimental results across all evaluated models.

---

## 1. Feature Engineering Pipeline (Pipeline A: 114 Handcrafted Image Features)

Implemented in [`PlantLeafFeatureExtractor`](../src/features/extractor.py) on `128 x 128` RGB leaf images:

| Feature Group | Count | Mathematical / Algorithmic Basis | Why It Matters for Plant Pathology |
| --- | ---: | --- | --- |
| **RGB Channel Moments** | 6 | Mean and standard deviation of normalized `R, G, B` in `[0, 1]` | Captures overall greenness vs. yellowing/browning shift |
| **HSV Moments + Histograms** | 30 | Normalized mean, std, and 8-bin histograms for `H, S, V` | Separates chromatic hue (`H`) from lighting brightness (`V`) |
| **CIELAB Moments + Histograms** | 30 | Normalized mean, std, and 8-bin histograms for `L*, a*, b*` | `a*` axis encodes green (`<128`) vs. red/brown (`>128`); `b*` encodes blue vs. yellow chlorosis |
| **Vegetation & Lesion Indices** | 8 | `ExG = 2G - R - B` (mean, std, `>0.05` ratio), `GRVI = (G-R)/(G+R+eps)` (mean, std), yellow/brown hue ratio (`H in [8,34], S>=40, V>=30`), dark necrotic ratio (`V in [15,75], S>=30`), `a* > 128` ratio | Directly encodes chlorophyll loss, yellowing halos, and dark necrotic lesions |
| **GLCM Texture Features** | 12 | `graycomatrix` at 64 gray levels, distances `d in {1, 3}`, averaged across 4 angles (`0, 45, 90, 135 deg`): `contrast, dissimilarity, homogeneity, energy, correlation, ASM` | Rotation-invariant surface roughness for mildew, rust, and leaf spot textures |
| **Uniform LBP Texture** | 18 | Rotation-invariant uniform Local Binary Patterns (`P=16, R=2`), normalized 18-bin histogram | Captures fine micro-texture stippling (e.g., spider mites, early spot rings) |
| **Hu Moments** | 7 | Log-signed Hu invariant moments on grayscale intensity | Scale/translation/rotation-invariant global shape descriptors |
| **Gradient & Edge Density** | 3 | Sobel gradient magnitude (mean, std) and Canny edge pixel ratio | Measures crisp lesion borders and vein prominence |
| **Total** | **114** | Zero-NaN float32 vector standardized via `StandardScaler` fit on `train` only | Interpretable tabular representation for Classical ML & Feature-MLP |

---

## 2. Model Architectures

### 2.1 Classical ML Baselines (on Standardized 114-Dim Features)

- **Logistic Regression:** Linear multinomial / binary classifier with `class_weight="balanced"`, `C=2.0`, `max_iter=1000`.
- **Random Forest:** 200-tree ensemble with `min_samples_leaf=2`, `class_weight="balanced_subsample"`, providing Gini feature importances.

### 2.2 Feature-MLP Neural Network (`FeatureMLP`)

Defined in [`src/models/classical_and_mlp.py`](../src/models/classical_and_mlp.py):
- `Linear(114 -> 256) + BatchNorm1d + ReLU + Dropout(0.30)`
- `Linear(256 -> 128) + BatchNorm1d + ReLU + Dropout(0.20)`
- `Linear(128 -> 64) + BatchNorm1d + ReLU + Dropout(0.15)`
- `Linear(64 -> C)` (`C = 2` for Binary, `C = 38` for Multi-Class)
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)` with `CosineAnnealingLR`, square-root inverse class weights, and early stopping (`patience=6`).

### 2.3 Custom CNN Built from Scratch (`PlantDiseaseCNN`)

Defined in [`src/models/custom_cnn.py`](../src/models/custom_cnn.py) (~1.21M parameters: `1,206,626` Binary / `1,211,270` 38-Class):
- **Input:** `3 x 224 x 224` ImageNet-normalized RGB tensor.
- **4 Convolutional Stages:** Channels `3 -> 32 -> 64 -> 128 -> 256`. Each stage contains two `3x3` `Conv2d(padding=1, bias=False)` + `BatchNorm2d` + `ReLU` layers, optional `Dropout2d` (`0.0, 0.05, 0.10, 0.10`), and `MaxPool2d(2)`.
- **Global Average Pooling (`AdaptiveAvgPool2d(1)`)** -> `256`-dim vector (avoids overfitting from huge flattened dense layers).
- **Classifier Head:** `Dropout(0.35) -> Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.245) -> Linear(128, C)`.

### 2.4 Transfer Learning CNN (`TransferPlantCNN` — MobileNetV2)

Defined in [`src/models/transfer_cnn.py`](../src/models/transfer_cnn.py) (`2,272,550` parameters):
- **Backbone:** Pretrained `torchvision.models.mobilenet_v2(weights=IMAGENET1K_V2)` depthwise-separable inverted residual blocks.
- **Head:** `Dropout(0.30) -> Linear(1280, 38)`.
- **Explainability:** [`GradCAM`](../src/models/gradcam.py) hooks onto the final convolutional block (`features[-1]`) to generate spatial attention heatmaps.

---

## 3. Training & Augmentation Configuration

- **Hardware & Precision:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), PyTorch `2.11.0+cu128` with Automatic Mixed Precision (`AMP`, `float16`).
- **Augmentations (`train` only):** `Resize(224x224)`, `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.3)`, `RandomRotation(20 deg)`, conservative `ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02)`.
- **Loss & Regularization:** Square-root inverse class weights $(N / n_c)^{0.5}$ normalized to mean 1.0; `label_smoothing=0.05` for 38-class CNNs; gradient clipping `max_norm=2.0`.

---

## 4. Quantitative Benchmark Results (Held-Out Leaf-Grouped Test Split: `7,825` Images)

All metrics are saved in `outputs/metrics/benchmark_summary.csv`.

### 4.1 Binary Classification: Healthy vs. Diseased

| Model | Input Representation | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 114 Handcrafted Features | 93.10% | 90.08% | 93.51% | 91.56% | 0.9833 |
| Random Forest | 114 Handcrafted Features | 97.78% | 97.42% | 96.87% | 97.14% | 0.9973 |
| Feature-MLP (NN) | 114 Handcrafted Features | 98.94% | 98.40% | 98.91% | 98.65% | 0.9993 |
| Custom CNN (Direct) | `224x224` Raw RGB Pixels | 99.49% | 99.33% | 99.36% | 99.35% | 0.9998 |
| Custom CNN [38-Class Rollup] | `224x224` Raw RGB Pixels | 99.71% | 99.66% | 99.59% | 99.62% | 0.99996 |
| **Transfer CNN (MobileNetV2) [Rollup]** | `224x224` Raw RGB Pixels | **99.95%** | **99.93%** | **99.93%** | **99.93%** | **0.99986** |

*Note on Rollup:* For 38-class CNNs, $P(\text{healthy}) = \sum_{c \in \text{healthy}} P(c)$ and $P(\text{diseased}) = 1 - P(\text{healthy})$. On `7,825` unseen test leaves, MobileNetV2 Rollup makes only **4 binary errors** (`2` healthy misclassified as diseased, `2` diseased misclassified as healthy).

### 4.2 Fine-Grained 38-Class Crop + Disease Classification

| Model | Parameters / Config | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 200 trees on 114 features | 91.55% | 88.75% | 86.91% | 87.53% | 91.43% |
| Logistic Regression | L2 linear on 114 features | 92.61% | 89.10% | 92.44% | 90.49% | 92.66% |
| Feature-MLP (NN) | 114 -> 256 -> 128 -> 64 -> 38 | 96.10% | 93.88% | 95.19% | 94.43% | 96.11% |
| Custom CNN (From Scratch) | 1,211,270 params (12 epochs) | 97.65% | 95.94% | 96.45% | 96.15% | 97.64% |
| **Transfer CNN (MobileNetV2)** | **2,272,550 params (8 epochs)** | **99.64%** | **99.52%** | **99.46%** | **99.49%** | **99.64%** |

---

## 5. Key Experimental Insights & Error Analysis

1. **Top Handcrafted Image Features:** Random Forest Gini importance ranks `veg_yellow_brown_ratio` (`0.1246` binary, `0.0237` multiclass), `hsv_h_hist_1`, `lab_b_hist_3`, `veg_exg_green_ratio`, `lbp_u_bin_*`, and `edge_canny_ratio` among the strongest predictors — confirming that chlorosis/lesion color shifts + leaf micro-texture carry high diagnostic signal.
2. **Why Multi-Class Rollup Beats Direct Binary Classification:** `Custom CNN [Rollup]` (`99.71%`) beats `Custom CNN (Direct Binary)` (`99.49%`) using the exact same architecture. Learning 38 crop+disease targets prevents pooling 26 unrelated pathologies together and respects crop-specific baseline leaf shades.
3. **38-Class Error Analysis (`outputs/metrics/error_analysis_transfer_cnn.json`):** Out of `7,825` test images, MobileNetV2 misclassified only **28 images** (`0.36%`). The most frequent confusions occur between visually similar fungal blights on the *same* crop:
   - `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` -> `Corn_(maize)___Northern_Leaf_Blight` (`6` errors; lowest class F1 = `0.9353`)
   - `Tomato___Early_blight` -> `Tomato___Septoria_leaf_spot` (`3` errors) or `Tomato___Late_blight` (`2` errors)
   - `Pepper,_bell___healthy` -> `Pepper,_bell___Bacterial_spot` (`2` errors)
   - `Tomato___Late_blight` -> `Tomato___healthy` (`2` errors)
