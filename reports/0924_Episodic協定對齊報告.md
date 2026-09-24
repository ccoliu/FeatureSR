# Episodic 協定對齊報告：與 CC-CDFSL / SOTA 論文對齊評估協定
最後更新：2026-09-24

## 背景：我們的主表數字無法與 SOTA 論文並列

整理相關論文時發現，CC-CDFSL（CVPR 2026）沿用 StepSPT（TPAMI 2025）的 source-free CDFSL 設定：**每個測試 episode 都從預訓練 CLIP 重新開始，只用該 episode 的 support set 微調，再分類該 episode 的 query**。同領域的 ATHA（arXiv 2605.29776）也是同樣的 episode-level 協定。

我們先前所有數字（`docs/FeatureSR_Methodology.md` §4 全部）用的是另一個協定：在固定的 strict few-shot pool 上訓練一次，評估時每個 episode 直接用 zero-shot 相似度分類，不使用該 episode 的 support（§4.1）。兩者差距很大，例如 5-shot CropDiseases：我們的 baseline 89.68%，論文的 CLIP-LoRA baseline 96.20%。**在這個差距解決前，任何架構改進都無法與 SOTA 比較。**

## 對齊項目

逐項比對 CC-CDFSL 論文（repo 內 `2603.17655v2.pdf`）與 CLIP-LoRA 官方程式碼（MaxZanella/CLIP-LoRA，`run_utils.py`、`lora.py`）：

| 項目 | CC-CDFSL / CLIP-LoRA | 我們原本 | 新實作 `train_episodic.py` |
|:---|:---|:---|:---|
| 協定 | 每個 episode 從預訓練 CLIP 重新微調 | 固定 pool 訓練一次 | ✅ 每個 episode 重置 LoRA |
| Episode 資料池 | BSCD-FSL 全量資料 | train/test 切分 | ✅ `bscd_mode=True` |
| LoRA 位置 | 視覺＋文字 encoder、q/k/v 各一組、所有層 | 只有視覺、q/k/v 合併＋out_proj | ✅ |
| LoRA 超參數 | r=2, α=1, dropout=0.25 | r=4, α=1, dropout=0 | ✅ |
| 優化器 | AdamW(lr=2e-4, wd=1e-2)、cosine→1e-6 | Adam(lr=1e-4/3e-5, wd=5e-4) | ✅ |
| CE 增強 | RandomResizedCrop(0.08–1)＋flip | 無 | ✅ |
| 評估 episode 數 | 1-shot 100、5-shot 400 | 相同 | ✅ |
| 每個 episode 訓練步數 | 論文寫「100 epochs」；CLIP-LoRA 官方為 500 × shots 步 | — | 見下節 |

LoRA 可訓練參數量 184,320，與 CLIP-LoRA 官方設定（r=2、雙 encoder、q/k/v）計算值一致。舊的 `train.py`、`evaluate.py` 與既有 checkpoint 不受影響（`CLIPLoRA` 的新選項預設值維持舊行為，已驗證舊 checkpoint 144 個 key 全部可載入）。

## 第一次嘗試失敗：「100 epochs」照字面解讀只有 100 步

第一版把論文的「fine-tuned for 100 epochs」照字面實作。但 5-way 下 support set 只有 5 張（1-shot）或 25 張（5-shot），都小於 batch size 32，**1 epoch = 1 步，整個微調只有 100 步**。結果全面低於論文：

| | 100 步 | 論文 CLIP-LoRA | 差距 |
|:---|:---:|:---:|:---:|
| EuroSAT 5-shot | 86.87 ± 0.72 | 92.63 | -5.8 |
| CropDiseases 5-shot | 80.61 ± 1.04 | 96.20 | -15.6 |
| ISIC 5-shot | 31.98 ± 0.55 | 50.68 | -18.7 |
| ChestX 5-shot | 22.19 ± 0.40 | 24.44 | -2.3 |
| EuroSAT 1-shot | 78.33 ± 1.81 | 81.49 | -3.2 |
| CropDiseases 1-shot | 71.68 ± 2.28 | 85.11 | -13.4 |
| ISIC 1-shot | 27.55 ± 1.02 | 35.23 | -7.7 |
| ChestX 1-shot | 21.75 ± 0.76 | 21.73 | 0.0 |

診斷依據是每個 episode 結束時的訓練 CE：ChestX 5-shot 1.57、ISIC 5-shot 1.24，接近 5-way 隨機猜測的 ln 5 ≈ 1.61，也就是模型連 25 張 support 都沒學起來。而且 5-shot 的 CE 比 1-shot 高、離論文的差距也比 1-shot 大，符合「步數固定、資料變多、更學不完」的模式。

## 步數診斷（配對比較）

ISIC、CropDiseases（差距最大的兩個）5-shot，各 20 個 episode。每個 episode 的亂數種子只由 (seed, episode 編號) 決定，所以三種步數跑的是**完全相同的 20 個 episode**，可做配對比較。腳本 `scripts/run_episodic_iters_diag.sh`，結果在 `results_episodic_diag/`。

| 5-shot | 100 步 | 500 步 | 2500 步 | 論文 |
|:---|:---:|:---:|:---:|:---:|
| ISIC | 31.87 ± 2.01 | 49.27 ± 3.65 | 48.93 ± 3.84 | 50.68 |
| CropDiseases | 77.00 ± 3.87 | 95.27 ± 1.97 | 97.07 ± 1.40 | 96.20 |
| 最終訓練 CE（中位數） | 0.56–1.28 | 0.007 | 0.000 | — |

配對差：
- 500 − 100 步：ISIC **+17.40 ± 3.87**、CropDiseases **+18.27 ± 2.71**。論文數字都落在 500 步的信賴區間內。
- 2500 − 500 步：ISIC -0.33 ± 2.11（無差異）、CropDiseases **+1.80 ± 1.01**（配對信賴區間不含 0，真實但小的提升）。

## 決定：統一 500 步

1-shot、5-shot 都用 500 步（`train_episodic.py --steps`，預設 500）。理由：
- 已足以在兩個差距最大的資料集上重現論文數字（信賴區間內）。
- 所有方法（baseline、CC-CDFSL、FeatureSR）使用同一個預算，內部比較是公平的。
- 成本：4090 上 5-shot baseline 每個資料集約 2.7 小時；若照 CLIP-LoRA 官方的 500 × shots（5-shot 2500 步），baseline 約 13 小時、加上 CC+FSR 約 40 小時，四個資料集完整比較要一到兩週。

**已知代價**：CropDiseases 5-shot 在 500 步下可能比官方設定低約 1.8pp。論文中必須明確寫出每個 episode 的步數，不能只寫「100 epochs」。

## 對既有結論的影響

在對齊協定下，**連純 CLIP-LoRA baseline 都高於我們之前所有 Feature-SR 的結果**（ISIC 約 49% vs 先前 FSR 的 43–45%；CropDiseases 約 95–97% vs 約 90%）。§4 所有 Feature-SR / Ensemble / TTA 的增益，全部是在較弱的舊協定下量到的，在新協定下是否仍然成立**完全未知**，必須重新驗證。舊協定數字仍可作為內部消融參考，但不能放進與 SOTA 並列的主表。

## 下一步

1. 用 500 步重跑 8 組純 CLIP-LoRA baseline（`scripts/run_episodic_baseline.sh`），確認 EuroSAT、ChestX 與 1-shot 也對齊。
2. 對齊確認後：+CC-CDFSL（Table 13 的 λ1/λ2）→ 確認能重現論文的 CC-CDFSL 增益。
3. 再加 FeatureSR，才是本論文貢獻的真正檢驗。

## 相關檔案

- 實作：`train_episodic.py`、`models/clip_lora.py`（新增 `encoder`/`qkv_mode`/`use_out_proj`/`reset_lora`）、`models/clip_wrapper.py`（`encode_text(with_grad=...)`）
- 100 步結果：`results_episodic/*_nofsr_nocc_loraboth_r2.*`、`logs/episodic_baseline/`
- 步數診斷：`scripts/run_episodic_iters_diag.sh`、`results_episodic_diag/`、`logs/episodic_iters_diag/`
- 500 步正式重跑：`scripts/run_episodic_baseline.sh` → `results_episodic/*_steps500.*`、`logs/episodic_baseline_steps500/`
