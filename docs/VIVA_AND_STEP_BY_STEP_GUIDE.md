# Viva Preparation & Step-by-Step Build Guide (From Scratch)

This document is designed specifically for **lab viva / oral examination**. It explains:
1. **How the project was built from scratch, step by step.**
2. **Why every single engineering and modeling choice was made** (and why alternatives were not chosen).
3. **Likely Viva Questions & crisp, accurate answers.**
4. **Honest Limitations & What Improvements Are Required Next.**

---

## Part 1: How We Built the Project From Scratch (Step-by-Step)

### Step 0: Isolated Environment & Reproducible Project Layout
- **What we did:** Created a Python virtual environment (`.venv`) inheriting system scientific packages, installed CUDA-enabled PyTorch (`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`) for the NVIDIA RTX 4050 GPU, centralized all hyperparameters/paths in `configs/config.yaml`, and enforced deterministic seeding (`seed = 42`) in [`src/utils/seed.py`](../src/utils/seed.py).
- **Why this step:** Reproducibility and clean modularity (`src/data`, `src/features`, `src/models`, `src/training`, `src/inference`, `app/`, `tests/`) prevent monolithic messy notebooks and allow each component to be unit-tested independently.

### Step 1: Dataset Acquisition (`src/data/download_dataset.py`)
- **What we did:** Sparse-cloned `spMohanty/PlantVillage-Dataset` from GitHub keeping `raw/color` (`54,305` RGB images across `38` classes) and `leaf_grouping/` (`leaf-map.json`), while excluding `raw/grayscale` and `raw/segmented`.
- **Why `raw/color` instead of `grayscale` or `segmented`?**
  - *Vs. `grayscale`:* Plant diseases show strong chromatic symptoms (yellow chlorosis, brown/black necrosis, orange rust, white mildew). Converting to grayscale throws away critical hue/saturation cues.
  - *Vs. `segmented`:* Using `raw/color` keeps natural leaf edges intact (avoiding segmentation mask artifacts on fine tendrils/edges) and lets us evaluate whether CNNs rely on the leaf or background via Grad-CAM. Using sparse Git checkout also saved >1.3 GB of disk/download bandwidth.

### Step 2: Data Auditing, MD5 Deduplication & Leaf-Grouped Splitting (`src/data/prepare_splits.py`)
- **What we did:**
  1. Verified all `54,305` files (`256x256` RGB, `0` corrupt files).
  2. Found and removed **`21` exact MD5 duplicate images**, leaving **`54,284` unique usable images**.
  3. Parsed every folder `Crop___Condition` into both `class_id` (`0..37`) and `binary_id` (`0 = healthy` for `15,071` images across `12` classes; `1 = diseased` for `39,213` images across `26` classes).
  4. Mapped images to physical leaves via `leaf_grouping/leaf-map.json` + filename normalization (`resolve_leaf_id` stripping `_final_masked` and `copy` suffixes), yielding **`20,015` unique physical leaf groups**.
  5. Used `StratifiedGroupKFold(n_splits=7)` grouped by `leaf_id` -> `Train` (`38,741` images / `14,298` leaves), `Val` (`7,718` / `2,858`), `Test` (`7,825` / `2,859`).
- **Why this step (CRITICAL VIVA POINT!):**
  - **Why not plain `train_test_split`?** PlantVillage contains multiple angles and `copy` variants of the *exact same physical leaf*. A naive random image split leaks near-identical views of the same leaf between Train and Test, inflating accuracy artificially. Grouping by `leaf_id` ensures **zero physical leaf overlap** across Train, Val, and Test.
  - **Why not split only into Train/Test?** Without a dedicated **Validation** split, tuning early stopping or LR schedules on Test would leak test performance into model selection.

### Step 3: Exploratory Data Analysis & Confounding Check (`docs/DATASET_AND_EDA.md`)
- **What we did:** Plotted binary, crop-level, and 38-class distributions (`outputs/figures/eda_*.png`) and inspected class/crop counts.
- **What we discovered & why it changed our design:**
  - There is a **2.60:1** binary imbalance (`72.2%` diseased vs `27.8%` healthy) and a **36.2:1** imbalance across 38 classes (`5,507` vs `152` images).
  - Crucially, **5 crops appear in only one health class** (`Blueberry`, `Raspberry`, `Soybean` are 100% healthy; `Orange`, `Squash` are 100% diseased). That insight drove our decision to train 38-class models and evaluate **rolled-up Healthy vs. Diseased** alongside direct binary models.

### Step 4: Handcrafted Computer Vision Feature Extraction (`src/features/extractor.py`)
- **What we did:** Extracted a **`114`-dimensional feature vector** per leaf image (`128x128` resize, multiprocess across all `54,284` images in ~2.5 min):
  - **Color (`66` dims):** RGB mean/std (`6`), HSV mean/std + 8-bin histograms (`30`), CIELAB mean/std + 8-bin histograms (`30`).
  - **Agricultural Vegetation & Lesion Indices (`8` dims):** Excess Green `ExG = 2G - R - B`, `GRVI = (G-R)/(G+R)`, yellow/brown chlorosis hue ratio, dark necrotic ratio, CIELAB `a* > 128` ratio.
  - **Texture (`30` dims):** Rotation-invariant GLCM (`contrast, dissimilarity, homogeneity, energy, correlation, ASM` at distances `1, 3` averaged over 4 angles = `12`) + rotation-invariant uniform LBP (`P=16, R=2` = `18`).
  - **Shape & Edges (`10` dims):** 7 log-signed Hu Moments + Sobel gradient mean/std + Canny edge density.
- **Why this step:**
  - The lab problem explicitly specifies *"using image features and neural networks."*
  - Extracting interpretable features lets us verify domain science (e.g., `veg_yellow_brown_ratio` emerged as the #1 Gini importance feature!) and build lightweight models that run on CPU in milliseconds.
  - **Why CIELAB + HSV in addition to RGB?** RGB mixes brightness and color; HSV isolates Hue, and CIELAB `a*` directly measures the green-to-red/brown axis while `b*` captures blue-to-yellow chlorosis.

### Step 5: Feature-Based Classical ML & Feature-MLP Neural Network (`src/training/train_classical_mlp.py`)
- **What we did:** Fit `StandardScaler` strictly on `X_train` (preventing scaling leakage), then trained three models on the exact same 114 features for both Binary and 38-Class:
  1. **Logistic Regression** (linear baseline: `93.10%` Binary / `92.61%` 38-Class).
  2. **Random Forest** (tree ensemble + feature importance: `97.78%` Binary / `91.55%` 38-Class).
  3. **Feature-MLP Neural Network** (`114 -> 256 -> 128 -> 64 -> C` with BatchNorm, ReLU, Dropout, AdamW, CosineAnnealingLR, early stopping: **`98.94%` Binary / `96.10%` 38-Class**).
- **Why compare all three on the same features?** It proves empirically that a non-linear neural network (`Feature-MLP`) captures cross-feature interactions (e.g., color shift *combined* with texture roughness) better than linear boundaries or axis-aligned tree splits.

### Step 6: Custom CNN Built From Scratch (`src/models/custom_cnn.py`, `src/training/train_cnn.py`)
- **What we did:** Built a 4-stage CNN (`3 -> 32 -> 64 -> 128 -> 256` double-Conv3x3 + BatchNorm + ReLU + Dropout2d + MaxPool2d + **Global Average Pooling** + FC head, ~1.21M params) and trained it from random weights on both Binary (`15` epochs -> **`99.49%` test acc**) and 38-Class (`12` epochs -> **`97.65%` test acc**, **`99.71%` rolled-up Binary acc**).
- **Why build a Custom CNN from scratch in addition to Transfer Learning?**
  1. Demonstrates first-principles mastery of convolution, BatchNorm, pooling, receptive fields, and regularization without relying solely on pretrained weights.
  2. **Why Global Average Pooling (`AdaptiveAvgPool2d(1)`) instead of `Flatten`?** Flattening a `256 x 14 x 14` feature map creates `50,176` inputs into the first FC layer (>6.4M weights on one layer!), causing severe overfitting. GAP reduces `256 x 14 x 14` to `256` spatial averages cleanly.
  3. **Why mild `ColorJitter(hue=0.02)` instead of strong hue jitter?** Large hue rotations would turn healthy green leaves yellow/brown and corrupt the label!

### Step 7: Transfer Learning CNN — MobileNetV2 (`src/models/transfer_cnn.py`)
- **What we did:** Fine-tuned `MobileNetV2` (`IMAGENET1K_V2`, `2.27M` params) end-to-end on 38 classes for `8` epochs with `AdamW(lr=5e-4)`, `CosineAnnealingLR`, `AMP (float16)`, square-root inverse class weights $(N / n_c)^{0.5}$, and `label_smoothing=0.05`.
- **Results:** **`99.64%` 38-class accuracy (`99.49%` Macro-F1)** and **`99.95%` rolled-up Binary accuracy (`0.9999` ROC-AUC)** on `7,825` unseen test leaves!
- **Why MobileNetV2 over ResNet50 or VGG16?**
  - Depthwise-separable convolutions give VGG/ResNet-level accuracy with only `2.27M` parameters (~9 MB checkpoint vs ~528 MB VGG16), making it practical for smartphone/edge agricultural deployment and fast to train on a 6 GB RTX 4050 GPU.
- **Why square-root inverse class weights $(N / n_c)^{0.5}$ instead of raw inverse weights $N / n_c$?**
  - With a 36:1 class imbalance, full $N / n_c$ weights over-amplify rare-class gradient noise; $(N / n_c)^{0.5}$ boosts minority recall (e.g., `Potato___healthy` = 152 images) smoothly without destabilizing optimization.

### Step 8: Unified Evaluation, Rolled-Up Binary Comparison & Grad-CAM (`src/training/evaluate_all.py`, `src/models/gradcam.py`)
- **What we did:** Evaluated all models on the exact same held-out test set, computed rolled-up Healthy vs. Diseased probabilities $P(\text{healthy}) = \sum_{c \in \text{healthy}} P(c)$, plotted ROC curves, 2x2 and 38x38 confusion matrices, error analysis (`28` total misclassifications out of `7,825`), and **Grad-CAM** overlays.
- **Why Grad-CAM?** Deep CNNs can cheat via background color or lighting artifacts ("Clever Hans" effect). Grad-CAM confirms visually (`outputs/figures/gradcam_showcase.png`) that MobileNetV2 focuses on rust pustules, early blight lesions, and leaf surface textures rather than background corners.

### Step 9: Unit Tests & Interactive Streamlit Demo (`tests/`, `src/inference/predictor.py`, `app/app.py`)
- **What we did:** Wrote `11` pytest unit tests verifying feature dimensions/invariants, model forward passes, Grad-CAM hooks, leaf-ID normalization, metrics, and predictor smoke tests (`11 passed`), plus a 3-tab Streamlit UI allowing live leaf upload, model switching, Grad-CAM rendering, and feature inspection.

---

## Part 2: Expected Viva Questions & Crisp Answers

1. **Q: Why did you extract both handcrafted image features AND train CNNs?**  
   **A:** The problem statement requires classifying leaf images *"using image features and neural networks."* Handcrafted features (color, vegetation indices, GLCM, LBP, Hu moments) + Feature-MLP give interpretability and fast CPU inference (`98.94%` binary / `96.10%` 38-class), while CNNs learn hierarchical spatial filters end-to-end (`99.64%` 38-class / `99.95%` rolled-up binary), allowing a fair empirical comparison.

2. **Q: Why does your 38-class CNN rolled up to binary (`99.71%` Custom, `99.95%` MobileNetV2) outperform the direct binary Custom CNN (`99.49%`)?**  
   **A:** Different crops have different healthy leaf colors and vein shapes, and 26 different diseases look completely different from one another. Collapsing them into 2 buckets forces a direct binary model to learn a messy union of unrelated visual patterns, whereas 38-class training gives explicit per-crop supervision and preserves species context before summing healthy probabilities.

3. **Q: How did you prevent data leakage in PlantVillage?**  
   **A:** Two ways: (1) Removed `21` exact MD5 duplicate files before splitting; (2) Used PlantVillage's `leaf_grouping/leaf-map.json` + filename normalization with `StratifiedGroupKFold` so multiple angles or `copy` images of the same physical leaf (`20,015` groups) never appear across Train, Val, and Test. Also, `StandardScaler` and class weights were computed strictly on `Train`.

4. **Q: Explain GLCM and LBP in one sentence each.**  
   **A:** **GLCM** (Gray-Level Co-occurrence Matrix) measures second-order spatial statistics of pixel intensity pairs (contrast, homogeneity, energy, correlation) to quantify surface roughness; **LBP** (Local Binary Patterns) thresholds each pixel's circular neighborhood against the center pixel into a rotation-invariant texture histogram.

5. **Q: Why did you convert images to HSV and CIELAB in addition to RGB?**  
   **A:** RGB couples illumination and color strongly. HSV separates pure hue (`H`) from lighting (`V`), and CIELAB's `a*` (green–red) and `b*` (blue–yellow) axes directly track chlorophyll loss, yellowing (chlorosis), and browning/necrosis.

6. **Q: What is the `ExG` index you implemented?**  
   **A:** Excess Green Index ($\text{ExG} = 2G - R - B$), a classic precision-agriculture metric that highlights healthy green chlorophyll foliage while suppressing reddish/brownish soil and necrotic lesions.

7. **Q: Why did `Feature-MLP` (`96.10%`) beat `Random Forest` (`91.55%`) on 38 classes?**  
   **A:** Decision trees split one feature orthogonally at a time, which struggles with 38 densely packed classes in continuous 114-D space; `Feature-MLP` learns smooth linear combinations and non-linear boundaries across correlated color and texture features simultaneously.

8. **Q: Why did you use `BatchNorm` and `Dropout` in both Feature-MLP and Custom CNN?**  
   **A:** `BatchNorm` stabilizes internal layer distributions and speeds up convergence; `Dropout` (and `Dropout2d` on feature maps) prevents co-adaptation of neurons/channels and reduces overfitting.

9. **Q: What is Global Average Pooling (`AdaptiveAvgPool2d`) and why prefer it over `Flatten`?**  
   **A:** GAP averages each spatial feature map (`14x14`) down to a single scalar per channel (`256`), eliminating millions of dense weights, reducing overfitting, and making the classifier less sensitive to exact spatial shifts.

10. **Q: Explain Transfer Learning and why MobileNetV2 converged in just 8 epochs (`99.64%`).**  
    **A:** MobileNetV2 already learned generic edge, color, blob, and texture filters on 1.2M ImageNet images. Fine-tuning transfers those low/mid-level filters to leaf lesions immediately, whereas a random-init Custom CNN had to learn edges from scratch over 12 epochs (`97.65%`).

11. **Q: What makes MobileNetV2 lightweight compared to ResNet or VGG?**  
    **A:** Depthwise separable convolutions (factoring standard convolution into spatial depthwise filtering + `1x1` pointwise mixing) and inverted residual linear bottlenecks cut parameters (`2.27M`) and FLOPs dramatically.

12. **Q: Why did you use `AdamW` + `CosineAnnealingLR` + `AMP`?**  
    **A:** `AdamW` decouples weight decay regularization from adaptive gradient updates; `CosineAnnealingLR` smoothly decays the learning rate toward zero for fine convergence; `AMP` (`float16` mixed precision) doubles throughput and cuts VRAM usage on the RTX 4050 Tensor Cores.

13. **Q: Why report Macro-F1 and ROC-AUC in addition to Accuracy?**  
    **A:** Because PlantVillage is imbalanced (`72.2%` diseased vs `27.8%` healthy; `5,507` vs `152` across 38 classes). Accuracy alone favors majority classes, whereas Macro-F1 weights all classes equally and ROC-AUC measures threshold-independent binary separability.

14. **Q: What is Grad-CAM and how does it work mathematically?**  
    **A:** Grad-CAM backpropagates the predicted class logit $y^c$ to the last convolutional feature maps $A^k$, global-average-pools the gradients $\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k}$, and computes $\text{ReLU}(\sum_k \alpha_k^c A^k)$ to highlight image regions positively supporting class $c$.

15. **Q: Which classes were hardest even for MobileNetV2, and why?**  
    **A:** Out of `7,825` test leaves, only `28` were misclassified — primarily `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` confused with `Corn_(maize)___Northern_Leaf_Blight` (`6` cases, F1 `0.9353`) and `Tomato___Early_blight` vs `Tomato___Septoria_leaf_spot` (`3` cases), because both are fungal foliar blights producing overlapping tan/brown lesions on the same host species.

---

## Part 3: Critical Limitations & Improvements Required Next

To demonstrate mature engineering thinking during viva, proactively acknowledge these **real limitations** and **concrete required improvements**:

1. **Controlled Background / Domain Shift (Biggest Real-World Limitation):**
   - *Problem:* PlantVillage images were captured in a lab by placing single detached leaves on uniform gray/black backgrounds. Models trained only on `raw/color` often drop substantially on real smartphone photos taken in fields with soil, overlapping leaves, shadows, and harsh sunlight.
   - *Required Improvement:* Train/fine-tune on in-the-wild field datasets (e.g., **PlantDoc**, **FGVC Plant Pathology**), apply background randomization / **CutMix** using `raw/segmented` leaf cutouts pasted onto natural agricultural backgrounds, and add stronger lighting/shadow augmentations.

2. **Species-Health Confounding in Binary Classification:**
   - *Problem:* Five crops in PlantVillage (`Soybean`, `Blueberry`, `Raspberry` = healthy only; `Orange`, `Squash` = diseased only) lack the opposite label. A binary classifier can partially cheat by recognizing species shape.
   - *Required Improvement:* Collect balanced healthy & diseased samples for all 14 crops, or enforce domain-adversarial / crop-stratified evaluation (holding out whole crop-health combinations or explicitly predicting crop + pathology in a dual-head multi-task architecture).

3. **Unmapped Leaf Groups in `leaf-map.json` (~24% of Images):**
   - *Problem:* While `41,111 / 54,305` images matched `leaf-map.json`, `15` classes had images without entries in `leaf-map.json` (falling back to filename-stem grouping). If two distinct filenames from those classes came from the same leaf, mild leakage could remain in those subset classes.
   - *Required Improvement:* Run perceptual hash (**pHash / CLIP embedding cosine similarity**) clustering across all images in unmapped classes to group near-duplicate photos before splitting.

4. **Closed-World Assumption (No "Non-Leaf / Unknown" Rejection):**
   - *Problem:* Softmax always assigns probabilities summing to 1.0 across the 38 known classes — uploading a car, hand, or an unseen plant species forces a confident false diagnosis.
   - *Required Improvement:* Add an **Out-of-Distribution (OOD)** detector (e.g., energy score / Mahalanobis distance / temperature-scaled max-logit threshold + leaf-vs-non-leaf binary pre-filter).

5. **Single-Disease, Image-Level Label Assumption:**
   - *Problem:* Real leaves can exhibit co-occurring infections (e.g., early blight + spider mites) and varying severity stages.
   - *Required Improvement:* Formulate disease detection as **multi-label classification** plus **lesion segmentation / severity percentage estimation** (mild / moderate / severe).

6. **Edge / Mobile Deployment Optimization:**
   - *Problem:* Serving a PyTorch model inside Streamlit requires a Python runtime (~GBs of dependencies).
   - *Required Improvement:* Export MobileNetV2 to **ONNX / TensorFlow Lite with INT8 post-training quantization** for <3 MB offline inference on Android/iOS farmer devices.
