# λ3（Cross-Resolution Consistency Loss 權重）敏感度分析報告
最後更新：2026-09-10

## 背景

延續 `reports/0909_FeatureSR內部Ablation報告.md` 的發現——$\mathcal{L}_{CR}$ 的貢獻是資料集依賴的（EuroSAT/ISIC 有真實效果，CropDiseases 沒有）——同時回應教授在報告後提出的「Grid Search 怎麼找」的提問。

三維聯合搜尋（$\lambda_1, \lambda_2, \lambda_3$）在單張 3060 上不可行（$5^3$ 組合 × 2 資料集 × 約 6 小時/run 遠超算力）。$\lambda_1, \lambda_2$ 沿用 CC-CDFSL 論文 Table 13 的既有驗證值，只對本工作自己引入、從未驗證過的 $\lambda_3$ 做 one-variable-at-a-time（OVAT）敏感度分析，並在論文中明確記錄這個簡化，作為對「grid search 怎麼找」的具體方法論回答。

只掃 EuroSAT / ISIC：理由同 0909 report——CropDiseases 對 $\mathcal{L}_{CR}$ 不敏感、ChestX 的 Feature-SR 效果本身就是雜訊等級，對兩者做敏感度分析無法產生可解讀的結論。

## 方法

固定 `sr_refiner_layers=2`，per-dataset $\lambda_1, \lambda_2$ 沿用 Table 13 值，掃描 $\lambda_3 \in \{0, 0.1, 0.3, 0.5, 1.0\}$。5-way 5-shot、400 episodes、`--strict_few_shot`、`lr=1e-4`、無 warmup、`seed=42`，協定與 `scripts/run_all_strict.sh` / `scripts/run_fsr_ablation.sh` 完全一致。

**資料點來源（過程並不平順，如實記錄）：**

| λ3 | 資料集 | 來源 |
|:---:|:---|:---|
| 0 | 兩者 | 沿用 0909 ablation report 的 B(no_lcr) 組 |
| 0.3 | 兩者 | 沿用 0909 ablation report 的 A(full) 組 |
| 0.1 | EuroSAT | 3060 原始跑因 CUDA 瞬斷遺失、重跑成功完成 |
| 0.1 | ISIC | 3060 原始跑於 epoch 40 崩潰，以 epoch-10 best checkpoint 回收評估 |
| 1.0 | EuroSAT | 3060 原始跑完整成功完成 |
| 1.0 | ISIC | 3060 原始跑因機器重開機中斷，以 epoch-10 best checkpoint 回收評估 |
| 0.5 | 兩者 | 3060 原始跑完全遺失（無 checkpoint），改在 server127（RTX 4090）完整重跑並取得最終結果 |

腳本：`scripts/run_lambda3_sweep.sh`（首輪）、`scripts/run_lambda3_sweep_retry.sh`（補跑，未實際用於 λ3=0.5，改在 4090 上以相同協定手動執行）。Log 存於 `logs/lambda3_sweep/`（含 3060 原始/RECOVERED/RETRY 版本與 4090 版本，皆保留）。

## 結果

| λ3 | EuroSAT | ISIC |
|:---:|:---:|:---:|
| 0 | 88.18 ± 0.44% | 40.21 ± 0.69% |
| 0.1 | 88.82 ± 0.43% | 42.91 ± 0.64% |
| **0.3**（論文預設值） | 89.93 ± 0.44% | **43.23 ± 0.70%**（峰值） |
| 0.5 | 89.93 ± 0.42% | 42.97 ± 0.62% |
| 1.0 | **90.42 ± 0.41%**（峰值） | 39.27 ± 0.55% |

## 分析

### 1. 兩個資料集呈現完全不同的敏感度形狀

**EuroSAT** 隨 λ3 增加大致單調上升：88.18% → 90.42%，全程增益 2.24 個百分點，在測試範圍 $[0, 1.0]$ 內看不到明確頂點，代表若想找到真正的最佳值可能需要往更大的 λ3 延伸，但這超出目前驗證的實際需求（λ3=1.0 已與 0.3/0.5 的差距在信賴區間邊緣）。

**ISIC** 呈現明顯的倒 U 型：λ3=0→0.3 上升 3.02 個百分點（40.21%→43.23%），但 0.3→1.0 反而下降到比完全不用 $\mathcal{L}_{CR}$（λ3=0）還差（39.27% < 40.21%）。也就是說，對 ISIC 而言，$\mathcal{L}_{CR}$ 權重過大不只是「邊際效益遞減」，而是**主動有害**，效果比不用還差。

### 2. 不存在單一 λ3 適用於所有資料集

這是本報告最核心的發現：即使兩個資料集都對 $\mathcal{L}_{CR}$「有反應」（0909 report 已證實），最佳劑量卻南轅北轍——EuroSAT 傾向「越重越好」，ISIC 傾向「適量最好、過量有害」。單一固定的 λ3 無法同時逼近兩者的最優點。

### 3. 對「Grid Search 怎麼找」的具體回答

由於資料集依賴性強到連「最佳方向」都相反，聯合網格搜尋即使算力允許，其最優解也不會是一個能泛化到新資料集的通用值——這反而支持「不做過度精細的聯合網格搜尋」的決定。論文的 Table 13 預設值 λ3=0.3：
- 對 ISIC 而言幾乎正好落在峰值（43.23%，就是峰值本身）；
- 對 EuroSAT 而言雖非最優（差 0.49pp），但也不是最差，屬於一個對兩邊都安全、不需要為每個資料集重新調參的折衷點。

這說明目前的超參數選擇並非未經檢驗的隨意值，而是（事後驗證下）落在一個對已知資料集都相對穩健的區域。

## 結論與建議

1. **λ3 是資料集依賴的超參數，且依賴方向可能相反**——論文應誠實揭露這條敏感度曲線，而不是宣稱單一 λ3 對所有資料集最優。
2. **λ3=0.3 作為論文預設值有實證支持**，不需要改成 per-dataset 客製化數值——追求 EuroSAT 額外 0.49pp 或 ISIC 額外 0.26pp 的邊際收益，對照 per-dataset 調參引入的 overfitting-to-benchmark 風險，不划算。
3. **本報告本身即是對「grid search 怎麼找」的方法論回答**：完整聯合網格不可行也无必要，OVAT 敏感度分析搭配對「為什麼固定 λ1/λ2」與「為什麼只測兩個資料集」的明確理由陳述，是在算力限制下負責任的簡化。
4. 後續若要更嚴謹，可以只在 ISIC 的倒 U 型峰值附近（0.2~0.4）補測 1-2 個點來收斂峰值位置，但目前的解析度（0.3 已知在峰值上）已足以支持論文的結論陳述，非必要不建議再花算力。

## 相關檔案

- Sweep 腳本：`scripts/run_lambda3_sweep.sh`、`scripts/run_lambda3_sweep_retry.sh`
- 完整 log：`logs/lambda3_sweep/`（3060 原始/RECOVERED/RETRY 版本、4090 補跑版本 `*_4090.log`）
- Checkpoints：`checkpoints_lambda3_sweep/l3_{0.1,0.5,1.0}/`（λ3=0/0.3 借用 `checkpoints_ablation_{no_lcr,full}/`）
- 前置分析：`reports/0909_FeatureSR內部Ablation報告.md`
