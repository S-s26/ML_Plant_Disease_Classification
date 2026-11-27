# Plant Disease Classification (ML Lab Mini-Project)

Classify plant leaf images into **Healthy vs. Diseased** categories (and **38 fine-grained crop–disease classes**) using **handcrafted computer vision image features** and **neural networks** on the [PlantVillage Dataset (`spMohanty/PlantVillage-Dataset`)](https://github.com/spMohanty/PlantVillage-Dataset).

---

## Key Highlights

- **Leakage-Free Data Engineering:** Audited `54,305` RGB images (`raw/color`), removed `21` exact MD5 duplicates (`54,284` usable images), and enforced **physical leaf-grouped splitting** (`StratifiedGroupKFold` over `20,015` leaf groups from `leaf_grouping/leaf-map.json`) — zero physical leaf overlap across Train (`38,741`), Validation (`7,718`), and Test (`7,825`).
- **Dual Modeling Paradigms:**
  1. **Pipeline A — Handcrafted Image Features (`114` dims):** RGB/HSV/CIELAB statistics & histograms, agricultural vegetation/lesion indices (`ExG`, `GRVI`, chlorosis & necrotic ratios), GLCM texture, rotation-invariant uniform LBP, Hu moments, and Sobel/Canny edges — evaluated with **Logistic Regression**, **Random Forest**, and a PyTorch **Feature-MLP Neural Network**.
  2. **Pipeline B — End-to-End Convolutional Neural Networks:** A **4-stage Custom CNN built from scratch** (`1.21M` params) and a fine-tuned **MobileNetV2 Transfer Learning CNN** (`2.27M` params) with **Grad-CAM** visual explanations.
- **Interactive Web App:** 3-tab **Streamlit** dashboard (`app/app.py`) for live leaf diagnosis, Grad-CAM heatmaps, handcrafted feature inspection, and benchmark plots.

---

## Held-Out Test Split Results (`7,825` Unseen Physical Leaves)

| Model | Task | Accuracy | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression (114 Features) | Binary | 93.10% | 91.56% | 0.9833 |
| Random Forest (114 Features) | Binary | 97.78% | 97.14% | 0.9973 |
| Feature-MLP Neural Network (114 Features) | Binary | 98.94% | 98.65% | 0.9993 |
| Custom CNN (Direct Binary) | Binary | 99.49% | 99.35% | 0.9998 |
| Custom CNN (From Scratch) [38-Class Rollup] | Binary | 99.71% | 99.62% | 0.99996 |
| **Transfer CNN (MobileNetV2) [38-Class Rollup]** | **Binary** | **99.95%** | **99.93%** | **0.99986** |
| Random Forest (114 Features) | 38-Class | 91.55% | 87.53% | — |
| Logistic Regression (114 Features) | 38-Class | 92.61% | 90.49% | — |
| Feature-MLP Neural Network (114 Features) | 38-Class | 96.10% | 94.43% | — |
| Custom CNN (From Scratch) | 38-Class | 97.65% | 96.15% | — |
| **Transfer CNN (MobileNetV2)** | **38-Class** | **99.64%** | **99.49%** | — |

---

## Project Structure

```text
MLProject/
├── configs/config.yaml                  # Paths, feature & training hyperparameters
├── data/
│   ├── raw/PlantVillage-Dataset/        # Sparse-cloned raw/color + leaf_grouping
│   └── processed/                       # dataset_splits.csv & handcrafted_features.npz
├── src/
│   ├── data/                            # Download, leaf-grouped splitting, PyTorch Dataset
│   ├── features/                        # 114-dim CV feature extractor & parallel builder
│   ├── models/                          # FeatureMLP, Custom CNN, Transfer CNN, Grad-CAM
│   ├── training/                        # Trainers & unified evaluation/plotting script
│   ├── inference/predictor.py           # Unified single-image inference + Grad-CAM
│   └── utils/                           # Seeding, config/JSON I/O, plotting helpers
├── app/app.py                           # Streamlit interactive diagnostic web app
├── scripts/run_pipeline.py              # End-to-end pipeline orchestrator CLI
├── tests/                               # Pytest unit & smoke tests (11/11 passing)
├── models/                              # Saved model checkpoints, scaler, class mapping
├── outputs/                             # Generated figures (EDA, ROC, CM, Grad-CAM) & metrics
└── docs/                                # Complete project & viva documentation
    ├── PROJECT_REPORT.md
    ├── DATASET_AND_EDA.md
    ├── ARCHITECTURE_AND_EXPERIMENTS.md
    ├── USER_GUIDE.md
    └── VIVA_AND_STEP_BY_STEP_GUIDE.md
```

---

## Quickstart

```powershell
# 1. Run all unit & smoke tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Launch the interactive Streamlit diagnostic app
.\.venv\Scripts\python.exe -m streamlit run app/app.py

# 3. (Optional) Reproduce the entire pipeline end-to-end
.\.venv\Scripts\python.exe scripts/run_pipeline.py --skip-download
```

---

## Documentation Index

- **[Project Report](docs/PROJECT_REPORT.md):** Full formal mini-project report (Abstract, Problem Statement, Methodology, Results, References).
- **[Viva & Step-by-Step Build Guide](docs/VIVA_AND_STEP_BY_STEP_GUIDE.md):** How the project was built from scratch, *why* each decision was made, 15+ viva Q&As, and required future improvements.
- **[Dataset & EDA](docs/DATASET_AND_EDA.md):** Data integrity audit, leaf-grouped split counts, crop confounding insight, and class breakdown.
- **[Architecture & Experiments](docs/ARCHITECTURE_AND_EXPERIMENTS.md):** Handcrafted feature breakdown, neural network architectures, training details, and error analysis.
- **[User Guide](docs/USER_GUIDE.md):** Installation, CLI commands, Streamlit walkthrough, and troubleshooting.
