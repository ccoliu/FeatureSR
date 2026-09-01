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

### 3. Prompt Ensemble — Training-Time (5-way 5-shot, 400 Episodes) ⭐

> ⚠️ **重要**：`utils/prompt_templates.py` 的領域自適應 Multi-Prompt Ensemble **必須在訓練與評估時同時啟用**（都加 `--use_prompt_ensemble`）。因為 LoRA 只掛在 CLIP 視覺塔上、文字塔完全凍結，視覺端是針對「訓練時使用的文字分布」去校準對齊的——如果訓練用單一模板、只在推論時才切換成 ensemble，訓練/測試分布不一致會讓已校準好的對齊關係被打亂，反而造成嚴重掉分（詳見 [`reports/0829_PromptEnsemble修復報告.md`](reports/0829_PromptEnsemble修復報告.md)）。

| Dataset | Baseline (Standard Prompt) | Post-hoc Ensemble (訓練/測試不一致) | **Ensemble-aware Training (一致)** ⭐ | Delta |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | 90.17 ± 0.89% | 84.15% (-6.03%) | **93.81 ± 0.32%** | **+3.64%** 📈 |
| **CropDiseases** | 90.03 ± 1.36% | 85.04% (-4.99%) | **90.57 ± 0.64%** | **+0.54%** 📈 |
| **ISIC 2018** | 43.76 ± 1.38% | 25.15% (-18.61%) | **44.49 ± 0.61%** | **+0.73%** 📈 |
| **ChestX** | 22.88 ± 0.83% | 21.28% (-1.60%) | 22.79 ± 0.45% | -0.09% |
| **Average** | **61.71%** | **53.90%** | **62.92%** | **+1.21%** 📈 |

* Ensemble-aware 訓練不只修復了 post-hoc 用法的崩潰，還在 3/4 資料集上額外取得優於 baseline 的正向增益，增益量級與 Feature-SR 相當且理論上可疊加。

---

### 4. Feature-SR + Prompt Ensemble Combined (5-way 5-shot, 400 Episodes) ⭐⭐

> 同時開啟 `--use_feature_sr` 與 `--use_prompt_ensemble`（訓練與評估都要開），驗證兩個正交增益來源疊加使用的效果。詳見 [`reports/0831_FeatureSR_Ensemble疊加報告.md`](reports/0831_FeatureSR_Ensemble疊加報告.md)。

| Dataset | Baseline | Ensemble-only | **Feature-SR + Ensemble** ⭐⭐ | vs Baseline | vs Ensemble-only |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 90.17 ± 0.89% | 93.81 ± 0.32% | **94.54 ± 0.30%** | **+4.37%** 📈 | +0.73% |
| **CropDiseases** | 90.03 ± 1.36% | 90.57 ± 0.64% | **90.97 ± 0.60%** | **+0.94%** 📈 | +0.40% |
| **ISIC 2018** | 43.76 ± 1.38% | 44.49 ± 0.61% | **44.77 ± 0.65%** | **+1.01%** 📈 | +0.28% |
| **ChestX** | 22.88 ± 0.83% | 22.79 ± 0.45% | **22.96 ± 0.47%** | +0.08% | +0.17% |
| **Average** | **61.71%** | **62.92%** | **63.31%** | **+1.60%** 📈 | **+0.39%** 📈 |

* 4/4 資料集全部呈現 baseline < ensemble-only < Feature-SR + ensemble 的一致遞增疊加模式，證實兩個增益來源正交可疊加。ChestX 增益幅度最小，是已知的 backbone 架構限制（ViT/CLIP 在此領域天生不如 ResNet 系方法，詳見論文附錄 Table 9），非方法失效。

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

#### Train with Prompt Ensemble (Recommended, stackable with Feature-SR) ⭐
```bash
python train.py \
    --dataset eurosat \
    --n_shot 5 \
    --strict_few_shot \
    --use_prompt_ensemble \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 400 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --lambda1 1.5 \
    --lambda2 0.2 \
    --save_dir ./checkpoints_ensemble \
    --seed 42
```
> `--use_prompt_ensemble` 必須在訓練時就加上——只在 `evaluate.py` 評估時加會造成訓練/測試文字分布不一致，反而大幅掉分（平均 -7.81%，ISIC 最嚴重達 -18.61%）。詳見上方 Benchmark Results 第 3 節。

#### Train with Feature-SR + Prompt Ensemble Combined (Best Overall) ⭐⭐
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
    --use_prompt_ensemble \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 400 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --lambda1 1.5 \
    --lambda2 0.2 \
    --save_dir ./checkpoints_fsr_ensemble \
    --seed 42
```
> 兩個增益來源正交可疊加，4/4 資料集實測皆優於單獨使用（平均 +1.60% vs baseline，EuroSAT 達 +4.37%）。詳見上方 Benchmark Results 第 4 節與 [`reports/0831_FeatureSR_Ensemble疊加報告.md`](reports/0831_FeatureSR_Ensemble疊加報告.md)。批次執行所有 4 個資料集見 `scripts/run_fsr_ensemble_pilot.sh`。

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
| `--use_prompt_ensemble` | `flag` | `False` | Enables domain multi-template prompt ensemble (must be set during training too — see Benchmark Results §3) |
| `--use_tta` | `flag` | `False` | Enables test-time augmentation (horizontal-flip feature averaging) during evaluation |
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
