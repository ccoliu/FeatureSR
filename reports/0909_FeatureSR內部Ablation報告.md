# FeatureSR 內部元件 Ablation 報告：FeatureRefiner 與 Cross-Resolution Consistency Loss
# 最後更新：2026-09-09

## 背景

在跟外部意見比對 feature upsampling 相關文獻(FSR-GAN、FeatUp、UPLiFT、ViT-Up，見 `docs/RelatedWork_FeatureUpsampling.md`)後發現：本工作的 $\mathcal{L}_{CR}$ 在結構上是 FeatUp reconstruction loss 的單視角特例，且因為只用單一視角（無 jitter），在資訊論上無法注入新的空間細節，只能作為錨定正則項。這使得「FeatureSR 的增益到底來自哪個元件」從「應該驗證」升級為「必須驗證」——如果 refiner 或 $\mathcal{L}_{CR}$ 被證明沒有貢獻，會直接影響論文能宣稱的 contribution 範圍。

## 驗證方法

在 `--use_feature_sr` 開啟的前提下，設計三組組態，逐一拿掉一個元件：

| 組態 | `sr_refiner_layers` | `lambda3` | 回答的問題 |
|:---|:---:|:---:|:---|
| A. full | 2 | 0.3 | 內部對照基準 |
| B. no_lcr | 2 | 0 | $\mathcal{L}_{CR}$ 是否有貢獻？ |
| C. no_refiner | 0 | 0.3 | FeatureRefiner（Transformer）是否有貢獻？ |

協定嚴格對齊 `scripts/run_all_strict.sh`（產生 methodology 文件 §4.2 Feature-SR standalone 結果的協定）：`lr=1e-4`、無 warmup、`--strict_few_shot`、per-dataset $\lambda_1,\lambda_2$（沿用論文 Table 13）、`seed=42`、不開 Prompt Ensemble。5-way 5-shot、400 episodes。

**A 組在本批次內重新訓練**，未借用 `docs/FeatureSR_Methodology.md` §4.2 的舊數字——`logs/strict/` 的既有結果與 §4.2 表格對不上（例如 EuroSAT 90.47% vs 90.16%、ISIC 42.76% vs 43.20%），顯示 §4.2 數字來自另一次獨立 run，provenance 不明，故用同批次的 A 組作為內部一致的基準，避免跨 run 比較的誤差。

**ChestX 未納入**：其 Feature-SR 增益本身就在雜訊等級（+0.03%，見 §4.2），對一個 null effect 做 ablation 無法產生可解讀的結論。

腳本：`scripts/run_fsr_ablation.sh`，log 存於 `logs/fsr_ablation/`，checkpoint 分別存於 `checkpoints_ablation_{full,no_lcr,no_refiner}/`。

## 結果

| 資料集 | A. full | B. no_lcr | Δ（$\mathcal{L}_{CR}$ 的貢獻） | C. no_refiner | Δ（Refiner 的貢獻） |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 89.93 ± 0.44% | 88.18 ± 0.44% | **-1.75%** ✅ 真實 | 89.96 ± 0.43% | +0.03% ❌ 無效 |
| **CropDiseases** | 89.56 ± 0.67% | 89.65 ± 0.69% | +0.09% ❌ 無效 | 90.23 ± 0.65% | +0.67%（誤差範圍內）❌ 無效 |
| **ISIC 2018** | 43.23 ± 0.70% | 40.21 ± 0.69% | **-3.02%** ✅ 真實（最大）| 43.06 ± 0.64% | -0.17% ❌ 無效 |

「真實」判定標準：$|\Delta|$ 超過兩邊 95% 信賴區間之和（例如 EuroSAT：$1.75\% > 0.44\%+0.44\%=0.88\%$；ISIC：$3.02\% > 0.70\%+0.69\%=1.39\%$），代表效果不是採樣雜訊。

## 分析

### 1. FeatureRefiner（2 層 Pre-LN Transformer，4.21M 參數）在三個資料集上一致無貢獻

C 組（no_refiner）相對 A 組的 Δ 分別是 +0.03%、+0.67%、-0.17%，**三個都落在信賴區間範圍內**，沒有一個是統計上站得住的效果。這是三個測試資料集一致的結論，不是單一資料集的個案。

Refiner 的自注意力機制（在 784 個 token 上做全域 MHSA）沒有為分類任務帶來可量測的額外資訊，即便它佔了 FeatureSR 新增可訓練參數的 41.6%（4.21M / 10.13M）。

### 2. $\mathcal{L}_{CR}$ 的貢獻是依資料集而定的，不是普遍必要或普遍多餘

B 組（no_lcr）相對 A 組的 Δ：EuroSAT -1.75%、ISIC -3.02% 都超出信賴區間，是真實效果；CropDiseases +0.09% 則完全在雜訊範圍內。

這跟本專案先前觀察到的其他「依資料集決定」現象是同一種模式（TTA 依資料集開關、見 `reports/0901_TTA驗證報告.md`；1-shot 下 Ensemble/Feature-SR 疊加效果依資料集劇烈分化、見 `reports/0905_1shot完整驗證報告.md`）。合理推測：$\mathcal{L}_{CR}$ 對「細粒度局部線索的判斷邊界容易被升採樣雜訊擾動」的資料集（ISIC 皮膚病灶紋理、EuroSAT 地表細部）更關鍵，而 CropDiseases 的判別特徵可能更穩健，不需要額外的語意錨定。

### 3. 對「FeatureSR 是否只是額外參數量」這個質疑的回應

Refiner 無效意味著：FeatureSR 實際發揮作用的部分，只有 FeatureUpsampler（bilinear 插值 + 殘差卷積校正 + 可學習位置編碼，5.92M 參數）加上依資料集決定的 $\mathcal{L}_{CR}$ 錨定（0 額外參數，只是一個 loss 項）。也就是說，**§4.2 觀察到的 Feature-SR 增益，不是來自額外的模型容量（Transformer refiner），而是來自「更密的升採樣網格」本身，加上必要時的語意錨定**。這直接排除了「增益單純來自砸更多參數」的疑慮，因為砸最多參數的那個元件（refiner）被證明沒用。

## 結論與建議

1. **FeatureRefiner 可以從架構中移除**，不影響（在 CropDiseases 上甚至略微提升）分類準確率，同時省下 4.21M 可訓練參數（FeatureSR 新增參數量減少 41.6%，從 10.13M 降至 5.92M）。
2. **$\mathcal{L}_{CR}$ 應該保留，但需要誠實揭露其資料集依賴性**——不能宣稱它是普遍必要的機制，論文中應明確列出「哪些資料集需要它」。
3. 建議後續：
   - 用「FeatureUpsampler-only + 依資料集決定的 $\mathcal{L}_{CR}$」重新產生一組 5-shot / 1-shot 的正式基準結果，取代目前 provenance 不明確的 §4.2 舊數字。
   - 更新 `docs/FeatureSR_Methodology.md` 的架構描述與圖示，將 FeatureRefiner 標註為「經 ablation 驗證非必要」的可選元件，而非核心元件。
   - 在 related work / limitation 中明確引用本報告，作為「不是單純堆參數」的直接證據。

## 相關檔案

- Ablation 腳本：`scripts/run_fsr_ablation.sh`
- 完整 log：`logs/fsr_ablation/{dataset}_{A_full,B_no_lcr,C_no_refiner}.log`
- Checkpoints：`checkpoints_ablation_full/`、`checkpoints_ablation_no_lcr/`、`checkpoints_ablation_no_refiner/`
- 文獻定位分析（本報告的直接動機來源）：`docs/RelatedWork_FeatureUpsampling.md`
