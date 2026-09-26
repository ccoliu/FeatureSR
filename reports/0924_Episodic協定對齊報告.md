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

## 500 步正式結果（8 組，2026-09-25）

| | 500 步 | 論文 CLIP-LoRA | 差距 | 最終訓練 CE |
|:---|:---:|:---:|:---:|:---:|
| EuroSAT 1-shot | 82.51 ± 1.68 | 81.49 | +1.02 | 0.001 |
| CropDiseases 1-shot | 81.73 ± 1.93 | 85.11 | -3.38 | 0.003 |
| ISIC 1-shot | 32.17 ± 1.46 | 35.23 | -3.06 | 0.002 |
| ChestX 1-shot | 22.57 ± 0.95 | 21.73 | +0.84 | 0.028 |
| EuroSAT 5-shot | 92.42 ± 0.40 | 92.63 | -0.21 | 0.004 |
| CropDiseases 5-shot | 95.54 ± 0.41 | 96.20 | -0.66 | 0.007 |
| ISIC 5-shot | 47.71 ± 0.74 | 50.68 | -2.97 | 0.009 |
| ChestX 5-shot | 25.36 ± 0.49 | 24.44 | +0.92 | 0.246 |
| 平均 | 1-shot 54.75 / 5-shot 65.26 | 55.89 / 65.99 | -1.14 / -0.73 | |

4090 上耗時：1-shot 約 11 秒/episode、5-shot 約 24 秒/episode。

**判讀**：
- EuroSAT、ChestX 兩個 shot 都在誤差內對上。CropDiseases 5-shot 的 -0.66 與步數診斷量到的「500 步比 2500 步少約 1.8」一致，屬於已知代價。
- **ISIC 兩個 shot 都系統性低約 3 分**，不是雜訊，而且在 ISIC 上增加步數沒有幫助（診斷時 2500 ≈ 500）。剩下的差異應來自論文未揭露的細節，最可疑的是 prompt 模板，以及 RandomResizedCrop 的 0.08 下限可能把位於中央的病灶裁掉。CropDiseases 1-shot 的 -3.38 約 1.75 倍 CI，屬於邊緣。
- 論文只報平均值不報 CI，其數字本身也有抽樣誤差，實際吻合程度比表面更好。

**決定**：不為了追 ISIC 的 3 分去調 baseline。在測試 episode 上反覆調設定直到對上論文數字，本身就是對 benchmark 的過擬合。論文中如實揭露此差距。更有意義的對齊檢驗是第二階段：我們實作的 CC-CDFSL 能否重現論文報告的**相對增益**。

## 第二階段：CC-CDFSL 增益無法重現（2026-09-26）

在 500 步協定下，以 Table 13 的 λ1/λ2 加上我們的 CC-CDFSL 實作（`--cc_impl legacy`）。與 baseline 逐 episode 配對的結果（使用者於 server127 計算，原始 jsonl 待 push 後核對）：

| | CC-CDFSL − baseline | 論文報告 | +FSR − CC-CDFSL | +FSR − baseline |
|:---|:---:|:---:|:---:|:---:|
| ISIC 1-shot | +0.04 ± 0.80 | +2.90 | +0.68 ± 0.85 | +0.72 ± 0.93 |
| EuroSAT 1-shot | +0.52 ± 0.60 | +4.58 | -0.28 ± 0.71 | +0.24 ± 0.74 |
| CropDiseases 1-shot | +0.17 ± 0.55 | +3.80 | +0.27 ± 0.73 | +0.44 ± 0.74 |
| ChestX 1-shot | -0.19 ± 0.58 | +0.48 | +0.41 ± 0.80 | +0.23 ± 0.76 |
| ISIC 5-shot | +0.27 ± 0.43 | +4.04 | +0.24 ± 0.40 | +0.51 ± 0.45 |

CC-CDFSL 的增益全部接近 0，遠低於論文。剩下的 5-shot（EuroSAT/CropDiseases/ChestX）已停止，不在已知有偏差的實作上繼續花算力。

### 原因：我們的實作與論文公式有三處不同

逐條比對論文 Eq. 3–15：

1. **缺少 Eq. 5 的可訓練 MLP**：論文把 patch 特徵先經 `ReLU(L'W1)W2` 轉到文字空間；舊版 `CyclicConsistencyLoss` 沒有任何參數。
2. **I-T-I 檢索範圍（Eq. 14）**：論文只在 anchor 所屬影像的增強空間內檢索；舊版在所有影像的增強 patch 中檢索。
3. **T-I-T 沒有影像端梯度**：Eq. 7–9 兩次檢索都是 argmax，最後 loss 只由文字特徵組成（`1 − sim(T_j, T[argmax])`）。以合成資料驗證梯度：T-I-T 對視覺端與 MLP 的梯度皆為 0，只有文字端有梯度。

**對過去所有舊協定實驗的影響**：舊 `train.py` 的文字塔凍結、文字特徵在 `no_grad` 下計算，因此 `L_TIT.requires_grad = False`，**λ1 在所有舊協定實驗中完全沒有作用**，log 中的 TIT 值（~0.10–0.14）只是被加進總 loss 的常數。我們一直稱為「CC-CDFSL」的東西，實際上只有 CE + λ2·I-T-I。先前 FeatureSR/$\mathcal{L}_{CR}$ 的量測數字本身不受影響，但凡是寫「以 CC-CDFSL 為基礎」的地方都需要修正。

### 照論文公式字面重寫：MLP 坍縮

新增 `PaperCyclicConsistencyLoss`（`--cc_impl paper`），照 Eq. 3–15 實作：可訓練 MLP、同圖視角內 I-T-I（排除 anchor 自己的視角，否則會檢索到自己）、字面 T-I-T。梯度驗證符合預期：T-I-T 只更新文字端，I-T-I 更新視覺端與 MLP。

但 MLP 只從 I-T-I 拿到梯度，而 I-T-I 的目標是讓 `MLP(原 patch)` 與 `MLP(增強 patch)` 相似，輸出常數就能平凡滿足。本機 ISIC 1-shot、500 步、3 個 episode 實測：MLP 輸出兩兩平均餘弦從初始 0.31 升到 **0.992–0.997**，I-T-I loss 降到 **0.000–0.002**，確認坍縮。

### 結論

| 照論文公式字面實作 | 結果 |
|:---|:---|
| T-I-T（Eq. 7–9） | 影像端與 MLP 無梯度 |
| I-T-I + MLP（Eq. 5, 13–15） | MLP 坍縮成常數輸出，loss 被平凡降到 0 |
| 官方程式碼（github.com/z-yaz/CC-CDFSL） | 只有 README 與海報，未釋出 |

**CC-CDFSL 無法從論文描述重現。** 論文的實際實作必然包含公式未寫的機制（例如 stop-gradient、soft retrieval 或其他訓練 MLP 的方式），我們只能猜測。`--cc_impl` 預設值維持 `legacy`，`paper` 僅用於重現坍縮現象。

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
