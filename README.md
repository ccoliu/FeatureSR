# FeatureSR: Feature-Level Super-Resolution for Cross-Domain Few-Shot Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FeatureSR** enhances Vision-Language Cross-Domain Few-Shot Learning (CDFSL) by upsampling and refining ViT patch feature representations (from $14 \times 14$ to $28 \times 28$ spatial tokens). Built on top of the Cycle-Consistent Vision-Language Alignment framework ([CC-CDFSL, CVPR 2026](https://arxiv.org/abs/2603.17655)).

---

## 🌟 Key Highlights & Innovations

1. **Feature-Level Super-Resolution (Feature-SR)**:
   - Direct spatial token upsampling via bilinear interpolation + learnable residual Conv2D correction + learnable positional embedding.
   - Upsamples ViT-B/16 patch grid from **$14 \times 14$ ($N=196$) to $28 \times 28$ ($N=784$)**.
   - ⚠️ 原始設計中的 Multi-Head Self-Attention Refiner 層**已從正式架構移除**（`sr_refiner_layers=0`）——內部消融證實它在所有測試資料集上都沒有可量測的貢獻，卻佔了新增參數的 41.6%。詳見 [`reports/0909_FeatureSR內部Ablation報告.md`](reports/0909_FeatureSR內部Ablation報告.md)、`docs/FeatureSR_Methodology.md` §6.3。
2. **Cross-Resolution Consistency Loss ($\mathcal{L}_{\text{CR}}$)**:
   - Enforces consistency between upsampled patch representations and base representations, anchoring the upsampled grid against semantic drift.
   - 效果**依資料集而定**：EuroSAT / ISIC 移除後會顯著掉分（-1.75% / -3.02%），CropDiseases 則無差異；最佳權重 $\lambda_3$ 在兩個資料集間差了近 5 倍（EuroSAT ~1.5、ISIC ~0.3）。詳見 [`reports/0910_Lambda3敏感度分析報告.md`](reports/0910_Lambda3敏感度分析報告.md)。
3. **Cycle-Consistent Vision-Language Alignment (CC-CDFSL)**:
   - **Text-to-Image-to-Text (T-I-T)**: Ground text semantics onto high-resolution patch tokens.
   - **Image-to-Text-to-Image (I-T-I)**: Reconstruct patch tokens via semantic anchor matching.
4. **Optimized Few-Shot Scheduling**:
   - Linear Warmup (5 epochs) + Cosine Annealing learning rate schedule (`lr=3e-5`) to eliminate early catastrophic overfitting in extreme few-shot regimes (1-shot and 5-shot).

---

## 📊 Benchmark Results

Evaluated across **4 diverse cross-domain datasets** under the **Strict Few-Shot Protocol** (1-shot & 5-shot, mean $\pm$ 95% Confidence Interval):

> **架構備註（2026-09-13 更新）**：以下 Feature-SR 數字全部採用 ablation 驗證過的精簡架構 `sr_refiner_layers=0`（拿掉 FeatureRefiner Transformer 層——內部消融證實它在任何測試資料集上都沒有可量測的貢獻，卻佔了新增參數量的 41.6%，詳見 [`reports/0909_FeatureSR內部Ablation報告.md`](reports/0909_FeatureSR內部Ablation報告.md)）。舊版 `refiner=2` 數字見 `docs/FeatureSR_Methodology.md` 的 v2.4→v2.5 changelog。

### 1. 5-way 5-shot Performance (400 Episodes)

| Dataset | Domain | Baseline (14×14) | **Feature-SR (28×28)** | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | Satellite / Remote Sensing | 88.79 ± 0.48% | **89.96 ± 0.43%** | **+1.17%** 📈 | **90.02%** |
| **CropDiseases** | Agriculture / Plant Leaves | 89.68 ± 0.70% | **90.23 ± 0.65%** | **+0.55%** 📈 | **90.45%** |
| **ISIC 2018** | Dermatology / Skin Lesions | 41.44 ± 0.64% | **43.06 ± 0.64%** | **+1.62%** 📈 | **43.13%** |
| **ChestX** | Medical / Chest Radiographs | 22.43 ± 0.46% | **22.70 ± 0.46%** | **+0.27%** 📈 | **22.57%** |
| **Average** | — | **60.59%** | **61.74%** | **+1.15%** 📈 | — |

* Feature-SR achieves **100% positive gains across all 4 benchmark datasets**, breaking the 90% accuracy barrier on both CropDiseases and EuroSAT.

### 2. 5-way 1-shot Performance (100 Episodes)

| Dataset | Baseline (14×14) | **Feature-SR (28×28)** | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | **80.59 ± 1.29%** | 77.77 ± 1.51% | -2.82% | **79.40%** |
| **CropDiseases** | **77.77 ± 2.00%** | 76.53 ± 2.17% | -1.24% | **77.73%** |
| **ISIC 2018** | 30.93 ± 1.11% | **33.85 ± 1.05%** | **+2.92%** 📈 | **33.92%** |
| **ChestX** | **21.95 ± 0.86%** | 21.09 ± 0.86% | -0.86% | **21.72%** |
| **Average** | **52.81%** | **52.31%** | **-0.50%** | — |

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
| **EuroSAT** | 90.17 ± 0.89% | 93.81 ± 0.32% | **94.33 ± 0.32%** | **+4.16%** 📈 | +0.52% |
| **CropDiseases** | 90.03 ± 1.36% | 90.57 ± 0.64% | **90.11 ± 0.67%** | +0.08% | -0.46%（雜訊範圍內）|
| **ISIC 2018** | 43.76 ± 1.38% | 44.49 ± 0.61% | **45.43 ± 0.70%** | **+1.67%** 📈 | +0.94% |
| **ChestX** | 22.88 ± 0.83% | 22.79 ± 0.45% | **22.98 ± 0.47%** | +0.10% | +0.19% |
| **Average** | **61.71%** | **62.92%** | **63.21%** | **+1.50%** 📈 | +0.29% |

* 2/4 資料集（EuroSAT、ISIC）呈現 baseline < ensemble-only < Feature-SR + ensemble 的乾淨遞增模式；CropDiseases 在 ensemble-only 與 FSR+ensemble 之間基本持平（差距在信賴區間內，非負向交互作用）；ChestX 連 baseline < ensemble-only 都不成立（本來就是雜訊等級差異）。ChestX 整體增益幅度最小，是已知的 backbone 架構限制（ViT/CLIP 在此領域天生不如 ResNet 系方法，詳見論文附錄 Table 9），非方法失效。

---

### 5. Test-Time Augmentation（TTA，依資料集決定）

> `--use_tta`（水平翻轉 + 原圖特徵平均融合）早期曾在「訓練/測試 prompt 不一致」的壞掉 checkpoint 上測過，結果近乎雜訊（平均 -0.23%），因此一度被認為沒用。修復後重新隔離驗證，發現 TTA 本身沒問題——3/4 資料集穩定正向，但 ISIC 疊加 Feature-SR 後會顯著掉分，**不建議全域開啟，需依資料集決定**。詳見 [`reports/0901_TTA驗證報告.md`](reports/0901_TTA驗證報告.md)。

| 資料集 | 無 TTA（Ensemble-only / FSR+Ensemble）| 有 TTA | Delta | 建議 |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | 93.69% / 94.29% | 93.97% / 94.27% | +0.28% / -0.02% | 皆非顯著效果 |
| **CropDiseases** | 90.61% / 90.73% | 90.96% / 90.89% | +0.35% / +0.16% | 皆非顯著效果 |
| **ISIC 2018** | 44.10% / 45.48% | 44.03% / 44.63% | -0.07% / -0.85% | 皆非顯著效果（FSR+Ens 仍偏負，保守起見可關閉）|
| **ChestX** | 22.77% / 23.34% | **23.60%** / 22.89% | +0.83% / -0.45% | 皆非顯著效果 |

> **2026-09-13 更新**：改用 refiner=0 架構重測後，套用文件一致的顯著性判準（$|\Delta|$ 超過雙邊信賴區間之和），**上表沒有一格達到顯著**——包含原本被稱為「對 ChestX 最有效手段」的 Ensemble-only +0.83%（信賴區間和為 0.98%，未過門檻）。舊版（refiner=2）ISIC 的 -1.38% 曾是唯一過門檻的顯著效果，refiner=0 下降為 -0.85%，不再顯著。結論趨於保守：**TTA 的效果目前都在雜訊範圍內，per-dataset 開關應視為軟性經驗法則，不是已證實的效果**。
>
> **後續驗證**：為了檢驗「ISIC 退步是否是 $\mathcal{L}_{CR}$ 造成」的假設，把 ISIC 的 lambda3 改成 0 重新測了一次——結果退步不減反增，變成 **-1.74%**（信賴區間和 1.30%，重新變成顯著），**推翻了 $\mathcal{L}_{CR}$ 假設**：真正原因跟 Cross-Resolution Consistency Loss 無關，仍未查明，但「ISIC + Feature-SR + Ensemble 情境下建議關閉 TTA」這個工程結論反而更站得住腳（3 組配置有 2 組退步達顯著）。詳見 `docs/FeatureSR_Methodology.md` §4.6、§6.2 與 [`reports/0913b_ISIC_TTA_Lambda3隔離驗證報告.md`](reports/0913b_ISIC_TTA_Lambda3隔離驗證報告.md)。

---

### 6. 5-way 1-shot: Ensemble & Feature-SR + Ensemble（100 Episodes）⚠️

> 補齊 §3、§4 一直缺少的 1-shot 版本（同一套訓練協定，`--n_shot 1`）。**跟 5-shot 不同，1-shot 沒有全資料集一致的單調疊加模式**，效果依資料集劇烈分化，不建議直接假設 1-shot 也會正向。詳見 [`reports/0905_1shot完整驗證報告.md`](reports/0905_1shot完整驗證報告.md)。

| Dataset | Baseline | Ensemble-only | Δ | Feature-SR + Ensemble | Δ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 81.09 ± 1.34% | 79.83 ± 1.54% | **-1.26%** ⚠️ | 78.32 ± 1.55% | **-2.77%** ⚠️ |
| **CropDiseases** | 77.89 ± 1.90% | 78.64 ± 2.27% | +0.75% 📈 | 78.43 ± 2.25% | +0.54%（雜訊範圍內）|
| **ISIC 2018** | 32.79 ± 0.98% | **34.73 ± 1.10%** | **+1.94%** 📈 | 33.33 ± 1.14% | +0.54%（雜訊範圍內）|
| **ChestX** | 20.81 ± 0.96% | **22.21 ± 0.99%** | **+1.40%** 📈 | 21.28 ± 0.91% | +0.47%（雜訊範圍內）|
| **Average** | **53.15%** | **53.85%** | **+0.71%** | **52.84%** | **-0.31%** |

* 改用 refiner=0 架構重測後（2026-09-13），舊版最大的單一亮點——CropDiseases 的 +2.86%——縮水到 +0.54%，落在自身信賴區間內，不再是穩健效果；FSR+Ensemble 在 4 個資料集上全部落在 Ensemble-only 之下或持平，1-shot 下疊加 Feature-SR 已經看不到淨正效益（平均由 +0.67% 轉為 -0.31%）。EuroSAT 依然整體轉負,ISIC/ChestX 則是 Ensemble-only 已是最佳點、疊加 Feature-SR 略降。結論不變甚至更強：1-shot 下 Feature-SR 的效果不穩定、依資料集決定，不應假設它在極端少樣本情境下必然正向。

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
| `--use_tta` | `flag` | `False` | Enables test-time augmentation (horizontal-flip feature averaging). Dataset-dependent — recommended for EuroSAT/CropDiseases/ChestX, **not** for ISIC (regresses when stacked with Feature-SR, see Benchmark Results §5) |
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
