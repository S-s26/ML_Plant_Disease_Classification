# Dataset Analysis & Exploratory Data Analysis (EDA)

## 1. Dataset Source & Overview

- **Source Repository:** [`spMohanty/PlantVillage-Dataset`](https://github.com/spMohanty/PlantVillage-Dataset)
- **Original Citation:** Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science.
- **Image Variant Selected:** `raw/color` (original RGB leaf photographs taken against controlled neutral lab backgrounds).
- **Total Files Scanned:** `54,305` image files across `38` class folders (`14` crop species, `12` healthy classes, `26` diseased classes).

---

## 2. Data Integrity, Deduplication & Leakage Prevention

Running [`src/data/prepare_splits.py`](../src/data/prepare_splits.py) produced the following verified integrity audit (saved in `outputs/metrics/dataset_summary.json`):

| Audit Metric | Count | Action Taken |
| --- | ---: | --- |
| Total files found in `raw/color/` | 54,305 | Scanned & header-verified |
| Corrupt / unreadable images | 0 | Verified with Pillow |
| Exact MD5 duplicate images | 21 | Deduplicated before splitting |
| **Usable unique images** | **54,284** | Included in dataset splits |
| Resolution across all images | `256 x 256` | Uniform RGB dimensions |

### Why Leaf-Grouped Splitting (`StratifiedGroupKFold`) Matters

In PlantVillage, multiple photographs (and rotated/copied variants such as `copy` / `copy 2`) often come from the **same physical leaf**. A naive random `train_test_split` scatters images of the same leaf into both training and test sets, causing data leakage and overly optimistic accuracy.

Using the official `leaf_grouping/leaf-map.json` (40,328 entries) combined with identifier normalization (`resolve_leaf_id` in [`src/data/prepare_splits.py`](../src/data/prepare_splits.py)), all `54,284` images map to **`20,015` unique physical leaf groups**:

| Split | Images | % of Total | Physical Leaf Groups | Classes Present |
| --- | ---: | ---: | ---: | ---: |
| **Train** | 38,741 | 71.37% | 14,298 | 38 / 38 |
| **Validation** | 7,718 | 14.22% | 2,858 | 38 / 38 |
| **Test** | 7,825 | 14.41% | 2,859 | 38 / 38 |
| **Total** | **54,284** | **100.00%** | **20,015** | **38 / 38** |

Runtime assertions verify **zero physical leaf-group intersection** across Train, Validation, and Test.

---

## 3. Class & Binary Distribution Findings

### Binary Category Balance (Healthy vs. Diseased)

- **Healthy:** `15,071` images (`27.76%`) across 12 healthy classes
- **Diseased:** `39,213` images (`72.24%`) across 26 disease classes
- **Class Imbalance Ratio:** ~2.60:1 (Diseased : Healthy) in binary mode, and ~36.2:1 between the largest 38-class category (`Orange___Haunglongbing_(Citrus_greening)`: `5,507`) and the smallest (`Potato___healthy`: `152`).

To prevent majority classes from dominating training, our models apply balanced weights (`class_weight="balanced"` for classical ML, square-root inverse class weights $(N / n_c)^{0.5}$ for neural networks).

### Crop-Level Distribution & Confounding Observation

| Crop Species | Healthy Images | Diseased Images | Total |
| --- | ---: | ---: | ---: |
| Apple | 1,638 | 1,526 | 3,164 |
| Blueberry | 1,502 | 0 | 1,502 |
| Cherry (including sour) | 854 | 1,052 | 1,906 |
| Corn (maize) | 1,162 | 2,690 | 3,852 |
| Grape | 423 | 3,639 | 4,062 |
| Orange | 0 | 5,507 | 5,507 |
| Peach | 360 | 2,297 | 2,657 |
| Pepper, bell | 1,478 | 997 | 2,475 |
| Potato | 152 | 2,000 | 2,152 |
| Raspberry | 371 | 0 | 371 |
| Soybean | 5,090 | 0 | 5,090 |
| Squash | 0 | 1,835 | 1,835 |
| Strawberry | 456 | 1,109 | 1,565 |
| Tomato | 1,585 | 16,561 | 18,146 |

**Critical EDA Insight:** Five crops (`Blueberry`, `Raspberry`, `Soybean` = healthy only; `Orange`, `Squash` = diseased only) have only one health state in PlantVillage. Therefore, a direct binary classifier can partially exploit leaf species shape rather than disease symptoms, whereas a 38-class model explicitly identifies crop + pathology and rolls predictions up to Healthy vs. Diseased.

---

## 4. Full 38-Class Split Breakdown

| Class Name | Binary Label | Train | Val | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| `Apple___Apple_scab` | diseased | 460 | 70 | 100 | 630 |
| `Apple___Black_rot` | diseased | 463 | 84 | 74 | 621 |
| `Apple___Cedar_apple_rust` | diseased | 188 | 44 | 43 | 275 |
| `Apple___healthy` | healthy | 1,200 | 224 | 214 | 1,638 |
| `Blueberry___healthy` | healthy | 1,064 | 235 | 203 | 1,502 |
| `Cherry_(including_sour)___Powdery_mildew` | diseased | 692 | 177 | 183 | 1,052 |
| `Cherry_(including_sour)___healthy` | healthy | 600 | 134 | 120 | 854 |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | diseased | 365 | 77 | 71 | 513 |
| `Corn_(maize)___Common_rust_` | diseased | 858 | 167 | 167 | 1,192 |
| `Corn_(maize)___Northern_Leaf_Blight` | diseased | 704 | 146 | 135 | 985 |
| `Corn_(maize)___healthy` | healthy | 826 | 168 | 168 | 1,162 |
| `Grape___Black_rot` | diseased | 852 | 168 | 160 | 1,180 |
| `Grape___Esca_(Black_Measles)` | diseased | 967 | 192 | 224 | 1,383 |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | diseased | 748 | 168 | 160 | 1,076 |
| `Grape___healthy` | healthy | 302 | 63 | 58 | 423 |
| `Orange___Haunglongbing_(Citrus_greening)` | diseased | 3,929 | 757 | 821 | 5,507 |
| `Peach___Bacterial_spot` | diseased | 1,664 | 311 | 322 | 2,297 |
| `Peach___healthy` | healthy | 243 | 60 | 57 | 360 |
| `Pepper,_bell___Bacterial_spot` | diseased | 727 | 133 | 137 | 997 |
| `Pepper,_bell___healthy` | healthy | 1,064 | 207 | 207 | 1,478 |
| `Potato___Early_blight` | diseased | 696 | 152 | 152 | 1,000 |
| `Potato___Late_blight` | diseased | 724 | 128 | 148 | 1,000 |
| `Potato___healthy` | healthy | 124 | 16 | 12 | 152 |
| `Raspberry___healthy` | healthy | 236 | 76 | 59 | 371 |
| `Soybean___healthy` | healthy | 3,688 | 709 | 693 | 5,090 |
| `Squash___Powdery_mildew` | diseased | 1,306 | 262 | 267 | 1,835 |
| `Strawberry___Leaf_scorch` | diseased | 777 | 147 | 185 | 1,109 |
| `Strawberry___healthy` | healthy | 330 | 64 | 62 | 456 |
| `Tomato___Bacterial_spot` | diseased | 1,524 | 295 | 308 | 2,127 |
| `Tomato___Early_blight` | diseased | 701 | 159 | 140 | 1,000 |
| `Tomato___Late_blight` | diseased | 1,356 | 281 | 264 | 1,901 |
| `Tomato___Leaf_Mold` | diseased | 700 | 108 | 144 | 952 |
| `Tomato___Septoria_leaf_spot` | diseased | 1,258 | 261 | 252 | 1,771 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | diseased | 1,184 | 244 | 248 | 1,676 |
| `Tomato___Target_Spot` | diseased | 1,009 | 199 | 196 | 1,404 |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | diseased | 3,819 | 752 | 786 | 5,357 |
| `Tomato___Tomato_mosaic_virus` | diseased | 269 | 53 | 51 | 373 |
| `Tomato___healthy` | healthy | 1,124 | 227 | 234 | 1,585 |

Generated EDA plots are stored in:
- `outputs/figures/eda_binary_and_crop_distribution.png`
- `outputs/figures/eda_38_class_distribution.png`
- `outputs/figures/eda_sample_leaves.png`
