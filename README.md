# FeatureSR: Feature-Level Super-Resolution for Cross-Domain Few-Shot Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FeatureSR** enhances Vision-Language Cross-Domain Few-Shot Learning (CDFSL) by upsampling and refining ViT patch feature representations (from $14 \times 14$ to $28 \times 28$ spatial tokens). Built on top of the Cycle-Consistent Vision-Language Alignment framework ([CC-CDFSL, CVPR 2026](https://arxiv.org/abs/2603.17655)).

---

## 🌟 Key Highlights & Innovations

1. **Feature-Level Super-Resolution (Feature-SR)**:
   - Direct spatial token upsampling via PixelShuffle-equivalent transposed convolution / bilinear projection + Multi-Head Self-Attention Refiner layers.
   - Upsamples ViT-B/16 patch grid from **$14 \times 14$ ($N=196$) to $28 \times 28$ ($N=784$)**.
2. **Cross-Resolution Consistency Loss ($\mathcal{L}_{\text{CR}}$)**:
   - Enforces consistency between upsampled patch representations and base representations, preventing feature distortion while enhancing fine-grained local alignment.
3. **Cycle-Consistent Vision-Language Alignment (CC-CDFSL)**:
   - **Text-to-Image-to-Text (T-I-T)**: Ground text semantics onto high-resolution patch tokens.
   - **Image-to-Text-to-Image (I-T-I)**: Reconstruct patch tokens via semantic anchor matching.
4. **Optimized Few-Shot Scheduling**:
   - Linear Warmup (5 epochs) + Cosine Annealing learning rate schedule (`lr=3e-5`) to eliminate early catastrophic overfitting in extreme few-shot regimes (1-shot and 5-shot).

---

## 📊 Benchmark Results

Evaluated across **4 diverse cross-domain datasets** under the **Strict Few-Shot Protocol** (1-shot & 5-shot, mean $\pm$ 95% Confidence Interval):

### 1. 5-way 5-shot Performance (400 Episodes)

| Dataset | Domain | Baseline (14×14) | **Feature-SR (28×28)** | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | Satellite / Remote Sensing | 88.79 ± 0.48% | **90.16 ± 0.40%** | **+1.37%** 📈 | **90.43%** |
| **CropDiseases** | Agriculture / Plant Leaves | 89.68 ± 0.70% | **90.61 ± 0.65%** | **+0.93%** 📈 | **90.41%** |
| **ISIC 2018** | Dermatology / Skin Lesions | 41.44 ± 0.64% | **43.20 ± 0.63%** | **+1.76%** 📈 | **43.30%** |
| **ChestX** | Medical / Chest Radiographs | 22.43 ± 0.46% | **22.46 ± 0.42%** | **+0.03%** 📈 | **22.64%** |
| **Average** | — | **60.59%** | **61.61%** | **+1.02%** 📈 | — |

* Feature-SR achieves **100% positive gains across all 4 benchmark datasets**, breaking the 90% accuracy barrier on both CropDiseases and EuroSAT.

### 2. 5-way 1-shot Performance (100 Episodes)

| Dataset | Baseline (14×14) | **Feature-SR (28×28)** | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | **80.59 ± 1.29%** | 80.29 ± 1.54% | -0.30% | **82.00%** |
| **CropDiseases** | **77.77 ± 2.00%** | 76.37 ± 2.22% | -1.40% | **80.00%** |
| **ISIC 2018** | 30.93 ± 1.11% | **32.63 ± 1.20%** | **+1.70%** 📈 | **32.91%** |
| **ChestX** | **21.95 ± 0.86%** | 21.51 ± 0.88% | -0.44% | **21.33%** |
| **Average** | **52.81%** | **52.70%** | **-0.11%** | — |

---

## 🛠️ Installation & Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/ccoliu/FeatureSR.git
cd FeatureSR
```

### 2. Install Dependencies
```bash
conda create -n featuresr python=3.10 -y
conda activate featuresr

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

Place downloaded datasets into the `data/` directory following this structure:

```
data/
├── CropDiseases/
│   └── New Plant Diseases Dataset(Augmented)/
│       └── train/
│           ├── Apple___Apple_scab/
│           └── ... (38 classes)
├── EuroSAT/
│   └── 2750/
│       ├── AnnualCrop/
│       ├── Forest/
│       └── ... (10 classes)
├── ISIC2018/
│   ├── ISIC2018_Task3_Training_Input/
│   └── ISIC2018_Task3_Training_GroundTruth/
│       └── ISIC2018_Task3_Training_GroundTruth.csv
└── ChestX-ray8/
    ├── images/
    │   └── images/ (*.png)
    └── Data_Entry_2017_v2020.csv
```

To auto-download EuroSAT or inspect paths:
```bash
python download_datasets.py --data_root ./data
```

---

## 🚀 Training & Usage

### 1. Quick Start: Single Experiment

#### Train Baseline (CC-CDFSL, 14×14)
```bash
python train.py \
    --dataset eurosat \
    --n_shot 5 \
    --strict_few_shot \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 400 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --lambda1 1.5 \
    --lambda2 0.2 \
    --seed 42
```

#### Train with Feature-SR (28×28 Up-sampled Representation) ⭐
```bash
python train.py \
    --dataset eurosat \
    --n_shot 5 \
    --strict_few_shot \
    --use_feature_sr \
    --sr_scale 2 \
    --sr_refiner_layers 2 \
    --sr_refiner_heads 8 \
    --lambda3 0.3 \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 400 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --lambda1 1.5 \
    --lambda2 0.2 \
    --seed 42
```

### 2. Batch Execution: Run All 16 Benchmark Experiments
Run the full 16-experiment test suite across all 4 datasets and both 1-shot and 5-shot settings:

```bash
bash scripts/run_all_optimized.sh
```

---

## ⚙️ Key Arguments & Specification (`train.py`)

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--dataset` | `str` | `eurosat` | Target dataset: `eurosat`, `crop_disease`, `isic`, `chestx` |
| `--n_shot` | `int` | `5` | Number of support samples per class (e.g. `1` or `5`) |
| `--n_way` | `int` | `5` | Number of classes per episode |
| `--n_query` | `int` | `15` | Number of query samples per class per episode |
| `--strict_few_shot`| `flag`| `False` | Enforces true few-shot sampling pool matching target domain |
| `--use_feature_sr` | `flag`| `False` | Enables the Feature-level Super-Resolution module (14×14 → 28×28) |
| `--sr_scale` | `int` | `2` | Spatial upsampling factor (2x scales tokens from 14×14 to 28×28) |
| `--sr_refiner_layers`| `int`| `2` | Number of Transformer Refiner blocks in Feature-SR module |
| `--sr_refiner_heads` | `int` | `8` | Number of attention heads in Transformer Refiner |
| `--lambda1` | `float` | `1.0` | Weight for Text-Image-Text (T-I-T) cyclic loss |
| `--lambda2` | `float` | `0.5` | Weight for Image-Text-Image (I-T-I) cyclic loss |
| `--lambda3` | `float` | `0.3` | Weight for Cross-Resolution Consistency loss ($\mathcal{L}_{\text{CR}}$) |
| `--lr` | `float` | `3e-5` | Peak learning rate for Adam optimizer |
| `--warmup_epochs` | `int` | `5` | Linear warmup epochs before Cosine Annealing decay |

---

## 📂 Project Structure

```
FeatureSR/
├── models/
│   ├── clip_wrapper.py       # CLIP ViT-B/16 feature extractor
│   ├── clip_lora.py          # PEFT CLIP-LoRA integration
│   └── feature_sr.py         # ⭐ Feature-SR module & CrossResolutionConsistencyLoss
├── losses/
│   └── cycle_consistency.py  # T-I-T and I-T-I cycle consistency loss
├── datasets/
│   ├── base_dataset.py       # Few-shot base dataset class
│   ├── eurosat.py            # EuroSAT loader
│   ├── crop_disease.py       # CropDiseases loader
│   ├── isic.py               # ISIC 2018 loader
│   └── chestx.py             # ChestX-ray8 loader
├── utils/
│   ├── few_shot_sampler.py   # N-way K-shot episodic sampler
│   ├── augmentation.py       # Data augmentations
│   └── metrics.py            # Accuracy & 95% Confidence Interval calculations
├── scripts/
│   ├── run_all_optimized.sh  # Master script for all 16 experiments
│   └── resume_optimized.sh   # Resume interrupted runs
├── reports/                  # Detailed benchmark reports
├── train.py                  # Main training entry point
├── evaluate.py               # Evaluation entry point
└── requirements.txt          # Package requirements
```

---
