# User & Deployment Guide

This guide covers environment setup, reproducing the training & evaluation pipeline, running unit tests, and launching the interactive Streamlit web application.

---

## 1. Prerequisites

- **OS:** Windows 10/11 (or Linux/macOS with minor path adjustments)
- **Python:** 3.11+ (tested on Python 3.13.5)
- **Git:** 2.35+ (required for sparse-checkout download from GitHub)
- **Hardware:** NVIDIA GPU with CUDA support recommended for fast CNN training (tested on NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM); CPU fallback is automatically supported.

---

## 2. Environment Setup

Create and activate the project virtual environment, then install dependencies:

```powershell
# From the repository root (MLProject/)
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1

# Install core packages
pip install -r requirements.txt

# Install CUDA-enabled PyTorch (adjust cu128/cu126 for your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch and GPU detection:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. Running the Full Pipeline

You can run the complete pipeline from dataset download through evaluation with a single command:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

If the dataset has already been downloaded into `data/raw/PlantVillage-Dataset`, skip the download step:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --skip-download
```

### Step-by-Step Execution

Each stage can also be executed independently:

1. **Download PlantVillage (`raw/color` + `leaf_grouping`) via sparse checkout:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.data.download_dataset
   ```
2. **Verify integrity, deduplicate exact MD5 matches, and create stratified 70/15/15 splits:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.data.prepare_splits
   ```
3. **Extract 114 handcrafted image features across all splits:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.features.build_feature_table
   ```
4. **Train Logistic Regression, Random Forest, and Feature-MLP on extracted features:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.training.train_classical_mlp
   ```
5. **Train Custom CNN (from scratch) on Binary and 38-Class tasks:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.training.train_cnn --arch custom --task binary
   .\.venv\Scripts\python.exe -m src.training.train_cnn --arch custom --task multiclass
   ```
6. **Train MobileNetV2 Transfer Learning CNN on 38-Class task:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.training.train_cnn --arch transfer --task multiclass
   ```
7. **Evaluate all models, compute rolled-up Healthy vs. Diseased metrics, and generate plots:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.training.evaluate_all
   ```

---

## 4. Launching the Interactive Web Demo (Streamlit)

Start the Streamlit web application from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

*(If `.venv` is activated in your terminal, `streamlit run app/app.py` or `python -m streamlit run app/app.py` also works.)*

Then open the local URL printed in the terminal (typically `http://localhost:8501`) in a web browser.

### Demo Tabs

- **Live Leaf Diagnosis:** Upload any JPG/PNG leaf photo or select a held-out test sample, switch between `Transfer CNN (mobilenet_v2)`, `Custom CNN (From Scratch)`, and `Feature-MLP (Handcrafted Features)`, view the Healthy vs. Diseased verdict, Top-5 class probabilities, Grad-CAM visual attention overlay, and extracted color/texture indicators.
- **Model Benchmark & Results:** Inspect test-split accuracy, macro-F1, ROC-AUC curves, confusion matrices, and top feature importances.
- **Dataset EDA & Grad-CAM:** Review dataset composition, sample leaf grids, and Grad-CAM test showcases.

---

## 5. Running Unit Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 6. Configuration

All hyperparameters, paths, image sizes, and split ratios are centralized in `configs/config.yaml`.

---

## 7. Troubleshooting

- **Out of CUDA memory (`CUDA out of memory`):** Reduce `training.batch_size` in `configs/config.yaml` from `64` to `32` or `16`.
- **Slow DataLoader on Windows:** Set `training.num_workers` in `configs/config.yaml` to `0` or `2` if multiprocessing workers hit memory limits.
- **Pretrained weights download fails offline:** Run once with internet access to cache `MobileNet_V2_Weights.IMAGENET1K_V2` into `~/.cache/torch/hub/checkpoints/`.
