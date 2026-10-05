# TransCLIP（transductive 推論）在 ATHA 強 baseline 上的驗證（2026-10-05）

教授建議嘗試 CC-CDFSL、ATHA 以外的方向後，第一個測試的方向。是本研究中**第一個在強 baseline 上通過「開發集選定、確認集驗證」的增益**。背景見 `reports/1001_研究進度總整理.md`。

## 方法

TransCLIP（Zanella et al., NeurIPS 2024 Spotlight；官方程式 github.com/MaxZanella/transduction-for-vlms）：以 GMM 加上 query 之間 kNN 圖的 Laplacian 項，並以初始預測為先驗，迭代更新 query 的類別分配。推論時同時使用該 episode 的 75 個 query（**不使用它們的標註**）與 support。模型與訓練完全是 ATHA baseline（CoOp ＋ LoRA r=16、logit scale 1-shot 2 / 5-shot 4、100 步、5 倍增強），TransCLIP 只作用在推論，輸入為 query / support 的 CLS 特徵與 CoOp 文字特徵。

實作：`scripts/transclip_dump.py` 直接呼叫官方的更新函式，在 `scripts/run_atha_dump.sh` 存下的特徵上逐 episode 執行；已驗證與官方 `TransCLIP_solver` 的最終預測 100% 一致（官方 `cls_acc` 與新版 numpy 不相容，只影響印出的準確率）。

## 評估流程

- 官方用另外切出的有標註驗證樣本，在 gamma ∈ {0.002, 0.01, 0.02, 0.2} 中挑選；episode 協定沒有額外標註，因此把 2 種初始化 × 4 個 gamma 共 8 個設定事先固定為候選。
  - 初始化 `clip100`：官方預設 softmax(100 · q·t)
  - 初始化 `model`：模型自己的校準 softmax(s · q·t)，s 為 ATHA 的 logit scale
- 開發集（seed 2，100 個 episode）依「配對差平均最大」選定一個設定，確認集（seed 1）只評估該設定。
- 其餘資料集**沿用 ISIC 選定的設定，不再調整**。
- 所有數字為同一批 episode、同一個模型的逐 episode 配對差 ± 95% CI。

## 結果

### ISIC：選定與確認

| ISIC，ATHA 有增強 | baseline | 選定設定 | − baseline |
|:---|:---:|:---:|:---:|
| 5-shot 開發集（n=100） | 53.49 | model4:0.002 | +0.29 ± 0.47（無增益，未進確認集） |
| 1-shot 開發集（n=100） | 37.73 | model2:0.002 | **+1.07 ± 0.75** |
| **1-shot 確認集（n=800）** | 37.86 | model2:0.002 | **+1.05 ± 0.25** |

1-shot 開發集上其他設定：model2 的 gamma 0.01 / 0.02 / 0.2 為 −0.65 / −1.32 / −1.04（皆顯著變差），clip100 為 +0.24 ~ −0.45。**對 gamma 很敏感，只有最小的 gamma 有增益。**

### 推廣到其他資料集（1-shot，seed 1，n=800，設定 model2:0.002）

| 1-shot | baseline | ＋TransCLIP | − baseline |
|:---|:---:|:---:|:---:|
| ISIC | 37.86 | 38.91 | **+1.05 ± 0.25** |
| EuroSAT | 83.05 | 86.13 | **+3.08 ± 0.23** |
| CropDiseases（Augmented 版）⚠️ | 84.18 | 89.38 | **+5.21 ± 0.34** |
| CropDiseases（BSCD 標準版，見下節） | 86.54 | 91.02 | **+4.48 ± 0.31** |
| ChestX | 21.80 | 21.86 | +0.06 ± 0.19 |
| 平均 | 56.72 | 59.07 | +2.35 |

（ATHA 論文報告的 1-shot：baseline 平均 55.92、ATHA 58.35。）

### CropDiseases 資料池驗證（1-shot，seed 1，n=800，設定 model2:0.002）

| 資料池 | 張數 | baseline | ＋TransCLIP | − baseline |
|:---|:---:|:---:|:---:|:---:|
| Kaggle Augmented 版（原本使用，含離線增強副本） | 70,295 | 84.18 | 89.38 | **+5.21 ± 0.34** |
| **BSCD-FSL 標準版**（Kaggle saroz014/plant-disease 的 dataset/train） | 43,456 | 86.54 | 91.02 | **+4.48 ± 0.31** |
| 檔名去重複版（從 Augmented 版濾掉增強副本） | 39,059 | 84.91 | 89.52 | **+4.61 ± 0.31** |

- **增益在標準版上大致成立**（+4.48），不是重複影像造成的假象。去重複版（+4.61）與標準版幾乎相同，兩者都比 Augmented 版低約 0.6–0.7，顯示 Augmented 版的離線增強副本只墊高了一小部分增益（三者資料池與 episode 不同，這個差距無法配對檢定）。
- **絕對數字不能互相代替**：標準版的 baseline 比 Augmented 版高約 2.4、比去重複版高約 1.6（ATHA 論文報告的 baseline 為 85.32）。兩者都來自 PlantVillage 但影像子集不同。我們先前所有 CropDiseases 的絕對數字都受資料池影響，**論文的最終表格應一律改用標準版**；只估計增益時，檔名去重複是可用的替代做法。
- 資料：`data/CropDiseases_bscd/dataset/train/`；清單 `scripts/make_crop_bscd_json.py` → `~/atha_data_bscd`；腳本 `scripts/run_transclip_crop_variants.sh`。

### 5-shot（seed 1，n=400，設定 model4:0.002，即 ISIC 5-shot 開發集依規則選出者）

| 5-shot | baseline | ＋TransCLIP | − baseline |
|:---|:---:|:---:|:---:|
| EuroSAT | 92.80 | 93.01 | **+0.21 ± 0.09** |
| ISIC | 55.52 | 55.73 | +0.21 ± 0.22 |
| CropDiseases ⚠️ | 96.49 | 96.76 | **+0.27 ± 0.11** |
| ChestX | 24.58 | 24.51 | −0.07 ± 0.22 |

- EuroSAT 5-shot 的增益統計上成立但很小（1-shot 為 +3.08）；ISIC 5-shot 在雜訊內，與開發集（+0.29 ± 0.47）一致。與 TransCLIP 論文「shot 越多增益越小」一致：**transductive 推論的增益集中在 1-shot**。
- ISIC 5-shot 的 baseline 與 10/1 的 `ISIC_5shot_base.jsonl` 逐 episode 完全相同（配對差 +0.00），確認 `--paired` 在不同次執行之間可重現。

## 判讀與限制

1. **協定不同**：transductive 推論只能與 transductive 方法比較（SF-CDFSL 中如 IM-DCL），不能放進 inductive 的 SOTA 表。
2. **CropDiseases 可能被高估**：我們用的是 Kaggle「Augmented」版（70,295 張，標準約 43,000），含同一原圖的離線增強副本。TransCLIP 利用 query 之間的 kNN 圖，近似重複的影像可能放大增益；需在 BSCD-FSL 標準版（PlantVillage 原始版）上重跑確認。ISIC（10,015）與 EuroSAT（27,000）的資料池與標準一致，不受此影響。
3. **ChestX 沒有增益**：baseline 接近隨機（5-way 20%），特徵幾乎不含類別資訊，query 之間的分群結構無法幫忙。
4. **增益集中在 1-shot**：ISIC 5-shot 開發集無增益，與 TransCLIP 論文「shot 越多增益越小」一致；5-shot 其他資料集尚未測。
5. **對 gamma 敏感**：論文中需報告所有候選的結果。

## 下一步

1. CropDiseases 換成 BSCD-FSL 標準版重跑（順便處理資料池問題）。
2. 5-shot：在開發集上用 5-shot 自己的候選選定設定，再推廣，確認增益是否只在 1-shot。
3. 實作 / 重現 transductive 比較對象（如 IM-DCL、LIMO），以及在我們的 CLIP-LoRA 框架上驗證 TransCLIP。

## 後續準備（10/5，GPU 跑 5-shot 時進行，尚未執行實驗）

### CropDiseases 去重複版（下一步第 1 點）

- BSCD-FSL 原始 benchmark 的 CropDiseases 是 Kaggle `saroz014/plant-disease`；我們（以及 ATHA 重現）用的是 `vipoooool/new-plant-diseases-dataset` 的 Augmented 版 train（70,295 張圖片；資料夾另有 437 個 Windows 下載留下的 `:Zone.Identifier` 附屬檔，loader 已略過）。
- Augmented 版的離線增強副本**可由檔名後綴辨認**：`_flipLR`、`_flipTB`、`_90deg`、`_180deg`、`_270deg`、`_new{N}degFlip{LR,TB}`（例如 `... 3003.JPG` 與 `... 3003_90deg.JPG`、`... 3003_new30degFlipLR.JPG`）。濾掉後剩 **39,059 張原圖**、38 類（每類 341–2,022 張），與標準版約 43,000 張接近，**不必重新下載**。
- 已準備好（未執行）：
  - `scripts/make_crop_dedup_json.py` → `~/atha_data_dedup/CropDiseases/novel.json`（從 ATHA 清單過濾，標籤不變）
  - `scripts/run_transclip_crop_dedup.sh`：去重複版 1-shot、seed 1、800 個 episode、鎖定設定 model2:0.002（約 2.3 小時）
- 若去重複後增益明顯縮小，表示原本的 +5.21 有一部分來自重複影像；這也會影響我們所有 CropDiseases 的絕對數字（是否全面換成去重複版，待決定）。

### Transductive 比較對象（下一步第 3 點）

| 方法 | 設定 | BSCD-FSL 結果 | 程式碼 | 適用性 |
|:---|:---|:---|:---|:---|
| IM-DCL（TIP） | SF-CDFSL，transductive 微調（information maximization ＋ distance-aware contrastive）；主要 backbone 為 ResNet10（miniImageNet 預訓練），另測 ViT-B/16 | 1-shot 平均 55.91（Crop 84.37 / EuroSAT 77.14 / ISIC 38.13 / ChestX 23.98）；5-shot 平均 66.72 | github.com/xuhuali-mxj/IM-DCL | 同一 benchmark 的 transductive 方法，可作為文獻比較；backbone 不同，數字不能直接並列 |
| LIMO（2025） | transductive few-shot CLIP：在 CLIP-LoRA（r=2、視覺＋文字、q/k/v，**與我們的 train_episodic.py 設定相同**）上加 query 的 information maximization 與對 zero-shot 預測的 KL | 只測 11 個一般資料集（含 EuroSAT），未測 ISIC / ChestX / CropDiseases；比 TransCLIP 平均高約 5%（4-shot） | github.com/ghassenbaklouti/LIMO | 可直接移植到我們的框架（每個 episode 微調時把 75 個 query 加進 loss），作為「transductive 微調」對「transductive 推論（TransCLIP）」的比較 |

## 相關檔案

| 內容 | 檔案 |
|:---|:---|
| 存特徵（ATHA） | `scripts/run_atha_dump.sh`（`SHOT`、`AUG`、`SEED`、`N`、`DS`）、`~/ATHA/coop_lora_trainer_local.py`（`--dump_dir`；差異見 `scripts/atha_local.patch`） |
| TransCLIP 離線評估 | `scripts/transclip_dump.py`（`--only 初始化:gamma`） |
| 推廣腳本 | `scripts/run_transclip_1shot_all.sh` |
| 結果 | `results_atha/transclip_*.txt`、`results_atha/*_base_seed*_n*.jsonl` |
| 特徵檔（repo 外） | `~/atha_dumps/{DS}_{K}shot[_noaug]_seed{S}/`（每個 episode 約 30–40 MB） |
