# LIMO（transductive 微調）報告（2026-10-06）

TransCLIP（transductive 推論）在 ATHA 強 baseline 上 1-shot 有效、5-shot 只有約 +0.2（`reports/1005_TransCLIP_transductive報告.md`），因此嘗試 transductive 微調：在每個 episode 微調時就把 query（不用標註）加進 loss。

## 方法

LIMO（Baklouti et al., 2025；官方程式 github.com/ghassenbaklouti/LIMO，clone 在 `~/LIMO`）：在 CLIP-LoRA 上加三項無標註損失（同官方 `limo.py`）：
- 條件熵（讓每個 query 的預測更確定），權重 1
- KL(zero-shot 預測 ‖ 目前預測)（不偏離原始 CLIP），權重 0.1
- KL(均勻分布 ‖ query 的平均預測)（避免全部預測成同一類），權重 10

權重一律用官方預設，未經調整。

## 實作

| | 我們的框架（`train_episodic.py --limo`） | ATHA（`~/ATHA/coop_lora_trainer_local.py --limo`） |
|:---|:---|:---|
| 模型 | CLIP-LoRA r=2、視覺＋文字、logit scale 100（與 LIMO 論文相同） | ATHA baseline（CoOp ＋ LoRA r=16、CE 的 logit scale 4、100 步、5 倍增強） |
| 每步的無標註樣本 | 從 75 個 query 隨機取 24 張，同樣做增強（標註部分仍是全部 support，與 baseline 的 CE 完全相同；官方為 batch 32、標註:無標註 = 1:3） | 從 query 的 5 個增強視角（375 張）隨機取，數量 = 每步 support 數（125，1:1） |
| zero-shot 參考 | 暫時把 LoRA 縮放係數設 0（已驗證輸出與原始 CLIP 完全相同） | 訓練前的模型（LoRA 的 B = 0、CoOp context 為初始值），事先算好 |
| 顯存 | — | CE 與 LIMO 損失分開反向傳播（梯度相加，數學上等價） |
| 回歸檢查 | 不加 `--limo` 時與 baseline 逐 episode 相同 | 同左 |

## 結果

### 我們的框架（5-shot，n=200，對照組為步驟 0 的 baseline，同一批 episode）

| 5-shot | baseline | ＋LIMO | − baseline |
|:---|:---:|:---:|:---:|
| ISIC | 50.32 | 53.64 | **+3.32 ± 0.85** |
| EuroSAT | 92.19 | 95.62 | **+3.43 ± 0.46** |

與 LIMO 論文（4-shot 比 CLIP-LoRA 平均 +2.8%）的量級一致；也遠大於 TransCLIP 在 5-shot 的約 +0.2。

### ATHA 強 baseline（ISIC 5-shot 開發集，seed 2，n=100）

ATHA 的 CE 在 logit scale 4、LIMO 官方在 100，因此事先固定兩個候選：

| ISIC 5-shot | 準確率 | − baseline |
|:---|:---:|:---:|
| ATHA baseline | 53.49 | — |
| ＋LIMO，損失用 scale 100（官方） | 35.23 | **−18.27 ± 1.76** |
| ＋LIMO，損失用 scale 4（同 ATHA 的 CE） | 54.15 | +0.65 ± 1.21 |

- scale 100：LIMO 損失的梯度遠大於 scale 4 的 CE，support 學不起來（訓練結束時 CE 仍約 1.3–1.4，5-way 隨機為 1.61）。
- scale 4：support 正常學起來（CE 約 0.05），但 ISIC 上沒有顯著增益。

### ATHA 強 baseline（EuroSAT 5-shot 開發集，seed 2，n=100，設定沿用 ISIC 選定的 scale 4）

| EuroSAT 5-shot | 準確率 | − baseline |
|:---|:---:|:---:|
| ATHA baseline | 92.75 | — |
| ＋LIMO（scale 4） | 96.04 | **+3.29 ± 0.63** |

**在強 baseline 的 5-shot 上有實質增益**，大小與我們框架上的 EuroSAT（+3.43）幾乎相同。效果依資料集而定（ISIC 無、EuroSAT 有），與 TransCLIP 的模式一致（ISIC 皆最小）。

### 確認集（seed 1，n=400，設定鎖定為 scale 4；`scripts/run_atha_limo_confirm.sh`）

| 5-shot，ATHA 強 baseline | baseline | ＋LIMO | − baseline |
|:---|:---:|:---:|:---:|
| **EuroSAT** | 92.80 | 95.95 | **+3.14 ± 0.33** |
| **CropDiseases（BSCD 標準版）** | 96.86 | 98.05 | **+1.19 ± 0.22** |
| ChestX | 24.58 | 24.20 | −0.38 ± 0.52（雜訊內） |

- EuroSAT 確認集與開發集（+3.29 ± 0.63）一致：**LIMO 在強 baseline 的 EuroSAT 5-shot 上的增益已確認。**
- CropDiseases baseline 已達 96.86%（剩餘錯誤率約 3.1%），LIMO 再減少約 38% 的錯誤。
- ChestX 無增益：baseline 接近隨機（5-way 為 20%），與 TransCLIP 在 ChestX 上的結果相同（1-shot +0.06、5-shot −0.07）。執行途中的暫定值一度為 −0.70 ± 0.65，最終 400 個 episode 落在雜訊內。

## 判讀

- **ISIC 上與局部分數相同的模式**：在標準 CLIP-LoRA 上有效（+3.3），到 ATHA 強 baseline 上消失；**但 EuroSAT 在強 baseline 上仍有 +3.29**，所以不是「強 baseline 吸收一切」，而是依資料集而定。
- 值得注意：我們的框架加上 LIMO 後，EuroSAT 5-shot 達 95.62，高於 ATHA baseline 的 92.80；ISIC 為 53.64，低於 ATHA 的 55.52（不同批 episode，僅供參考）。也就是說，**LIMO 讓簡單的 CLIP-LoRA 接近甚至超過 ATHA 的強 baseline**，但兩者的好處不疊加（至少在 ISIC 上）。
- **與 TransCLIP 互補**：TransCLIP（transductive 推論）的增益集中在 1-shot（5-shot 只有約 +0.2）；LIMO（transductive 微調）在 5-shot 上仍有 +1.2 ~ +3.1。兩者在 ISIC（5-shot）與 ChestX 上都無增益，有增益的資料集也相同（EuroSAT、CropDiseases）。
- ISIC 5-shot 依流程只跑了開發集（+0.65 ± 1.21，未進確認集）。

### 強 baseline 上的 transductive 結果總覽（皆為確認集，seed 1）

| ATHA 強 baseline | ISIC | EuroSAT | CropDiseases（BSCD 標準版） | ChestX |
|:---|:---:|:---:|:---:|:---:|
| 1-shot ＋TransCLIP（n=800） | **+1.05 ± 0.25** | **+3.08 ± 0.23** | **+4.48 ± 0.31** | +0.06 ± 0.19 |
| 5-shot ＋TransCLIP（n=400） | +0.21 ± 0.22 | **+0.21 ± 0.09** | （僅 Augmented 版：+0.27 ± 0.11） | −0.07 ± 0.22 |
| 5-shot ＋LIMO（n=400） | （開發集 +0.65 ± 1.21） | **+3.14 ± 0.33** | **+1.19 ± 0.22** | −0.38 ± 0.52 |

## 相關檔案

| 內容 | 檔案 |
|:---|:---|
| 我們的框架 | `train_episodic.py`（`--limo` 等）、`scripts/run_limo_5shot.sh`、`results_episodic/*_limo24.jsonl`、`logs/limo/` |
| ATHA | `~/ATHA/coop_lora_trainer_local.py`（差異見 `scripts/atha_local.patch`）、`scripts/run_atha_limo_dev.sh`、`scripts/run_atha_limo_eurosat.sh`、`results_atha/{ISIC,EuroSAT}_5shot_*_seed2_n100.jsonl`、`logs/atha_limo/` |
| ATHA 確認集 | `scripts/run_atha_limo_confirm.sh`、`results_atha/{EuroSAT,CropDiseases_bscd,ChestX}_5shot_{base,limo_s0}_seed1_n400.jsonl`、`logs/atha_limo_confirm_driver.log` |
