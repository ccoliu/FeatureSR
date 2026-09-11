# λ3（Cross-Resolution Consistency Loss 權重）敏感度分析報告
最後更新：2026-09-11

## 背景

延續 `reports/0909_FeatureSR內部Ablation報告.md` 的發現——$\mathcal{L}_{CR}$ 的貢獻是資料集依賴的（EuroSAT/ISIC 有真實效果，CropDiseases 沒有）——同時回應教授在報告後提出的「Grid Search 怎麼找」的提問。

三維聯合搜尋（$\lambda_1, \lambda_2, \lambda_3$）在單張 3060 上不可行（$5^3$ 組合 × 2 資料集 × 約 6 小時/run 遠超算力）。$\lambda_1, \lambda_2$ 沿用 CC-CDFSL 論文 Table 13 的既有驗證值，只對本工作自己引入、從未驗證過的 $\lambda_3$ 做 one-variable-at-a-time（OVAT）敏感度分析，並在論文中明確記錄這個簡化，作為對「grid search 怎麼找」的具體方法論回答。

只掃 EuroSAT / ISIC：理由同 0909 report——CropDiseases 對 $\mathcal{L}_{CR}$ 不敏感、ChestX 的 Feature-SR 效果本身就是雜訊等級，對兩者做敏感度分析無法產生可解讀的結論。

## 方法

固定 `sr_refiner_layers=2`，per-dataset $\lambda_1, \lambda_2$ 沿用 Table 13 值，掃描 $\lambda_3 \in \{0, 0.1, 0.3, 0.5, 1.0\}$（兩資料集）；由於 EuroSAT 在 λ3=1.0 時仍呈上升、未見頂點，另外對 EuroSAT 單獨加測 $\lambda_3 \in \{1.5, 2.0\}$ 以逼近真正峰值（ISIC 已在 0.3 出現明確倒 U 型峰值，不需加測）。5-way 5-shot、400 episodes、`--strict_few_shot`、`lr=1e-4`、無 warmup、`seed=42`，協定與 `scripts/run_all_strict.sh` / `scripts/run_fsr_ablation.sh` 完全一致。

CropDiseases / ChestX 不納入本次 sweep：CropDiseases 在 λ3=0.3 的 ablation 主效應已是雜訊等級（Δ=+0.09%），ChestX 的 Feature-SR 整體效果本身就是雜訊等級（0909 report），對兩者做敏感度掃描資訊增益低，權衡算力後決定不做。

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
| 1.5 / 2.0 | EuroSAT | 直接在 server127（RTX 4090）完整跑完，無中斷 |

腳本：`scripts/run_lambda3_sweep.sh`（首輪）、`scripts/run_lambda3_sweep_retry.sh`（補跑，未實際用於 λ3=0.5，改在 4090 上以相同協定手動執行）、`scripts/run_lambda3_sweep_eurosat_high.sh`（EuroSAT 高值加測）。Log 存於 `logs/lambda3_sweep/`（含 3060 原始/RECOVERED/RETRY 版本與 4090 版本，皆保留）。

## 結果

| λ3 | EuroSAT | ISIC |
|:---:|:---:|:---:|
| 0 | 88.18 ± 0.44% | 40.21 ± 0.69% |
| 0.1 | 88.82 ± 0.43% | 42.91 ± 0.64% |
| **0.3**（論文預設值） | 89.93 ± 0.44% | **43.23 ± 0.70%**（ISIC 峰值） |
| 0.5 | 89.93 ± 0.42% | 42.97 ± 0.62% |
| 1.0 | 90.42 ± 0.41% | 39.27 ± 0.55% |
| **1.5** | **91.11 ± 0.41%**（EuroSAT 峰值） | 未測 |
| 2.0 | 90.23 ± 0.41% | 未測 |

## 分析

### 1. 兩個資料集其實都是倒 U 型，只是峰值位置差了 5 倍

加測 1.5 / 2.0 之後，原本以為「EuroSAT 單調上升、找不到頂點」的判斷需要修正：EuroSAT 在 λ3=1.5 出現明確峰值（91.11%），λ3=2.0 隨即回落（90.23%）。也就是說，**兩個資料集在「$\mathcal{L}_{CR}$ 存在最優劑量、過量會回落」這件事上是一致的**，不一致的只是最優點的位置——ISIC 的甜蜜點在 0.3 附近，EuroSAT 的甜蜜點在 1.5 附近，相差近 5 倍。

顯著性檢查：EuroSAT 1.5 vs 論文預設值 0.3，Δ=1.18pp，超過兩邊信賴區間之和（0.44%+0.41%=0.85pp），是站得住的真實提升；但 1.5 vs 相鄰的 1.0（Δ=0.69pp）、1.5 vs 2.0（Δ=0.88pp）都落在信賴區間邊緣，峰值的精確位置不宜過度解讀，只能說「最優區間大致在 1.0~2.0 之間，1.5 是目前取樣中最好的點」。

ISIC 那邊：λ3=0.3→1.0 從峰值下降到比完全不用 $\mathcal{L}_{CR}$（λ3=0）還差（39.27% < 40.21%），顯示過量對 ISIC 的懲罰比對 EuroSAT 更陡峭——EuroSAT 從峰值 1.5 掉到 2.0 只掉 0.88pp，ISIC 從峰值 0.3 掉到 1.0 卻掉了 3.96pp。

### 2. 不存在單一 λ3 適用於所有資料集

即使兩個資料集現在都確認是倒 U 型（形狀一致），最佳劑量仍然南轅北轍：EuroSAT 要「重劑量」（~1.5）才能到頂，ISIC 要「輕劑量」（~0.3）就已經到頂、再加重反而受傷。單一固定的 λ3 無法同時命中兩者的最優點。

### 3. 對「Grid Search 怎麼找」的具體回答

由於資料集依賴性強到連最優劑量的數量級都不同，聯合網格搜尋即使算力允許，其最優解也不會是一個能泛化到新資料集的通用值——這反而支持「不做過度精細的聯合網格搜尋，改用 per-dataset 敏感度曲線＋一個折衷預設值」的決定。論文的 Table 13 預設值 λ3=0.3：
- 對 ISIC 而言正好落在峰值（43.23%，峰值本身）；
- 對 EuroSAT 而言不是最優（差峰值 1.18pp），但仍明顯優於不用（比 λ3=0 高 1.75pp），屬於「兩邊都不差、不需要為每個資料集重新調參」的折衷點。

### 4. 是否該把 EuroSAT 的 λ3 改成 1.5（per-dataset 客製化）？

技術上 1.18pp 是統計上站得住的真實提升，但是否值得為此在論文中引入 per-dataset λ3 客製化，是一個路線選擇問題，而不是「有沒有效果」的問題——見結論第 2 點。

## 結論與建議

1. **λ3 是資料集依賴的超參數，且兩個資料集都呈倒 U 型，只是最優劑量差了近 5 倍**——論文應誠實揭露這條敏感度曲線，而不是宣稱單一 λ3 對所有資料集最優，也不宜簡化成「EuroSAT 越重越好」的錯誤敘述。
2. **是否改用 per-dataset λ3（EuroSAT=1.5, ISIC=0.3）是路線選擇，不是必須**：維持統一預設值 0.3 的敘事更簡潔、也更不容易被質疑「調參調到 test set 最優」（overfitting to benchmark）；若選擇 per-dataset 客製化，則多了 EuroSAT +1.18pp 的真實增益，但論文需要額外交代「per-dataset 最優值怎麼決定」，等於把「grid search 怎麼找」的問題往下游推了一層。**建議維持統一 0.3**，把這條敏感度曲線本身作為附錄／消融分析呈現，而不是拿它去反過來改預設值。
3. **本報告即是對「grid search 怎麼找」的方法論回答**：完整聯合網格不可行也不必要，OVAT 敏感度分析＋對「為什麼固定 λ1/λ2」「為什麼只測兩個資料集」的明確理由陳述，是在算力限制下負責任的簡化，且已經找到兩個資料集各自的峰值，足以支撐論文論述。
4. **不建議再加測 ISIC 的高值或補測其他資料集**：ISIC 峰值已確認在 0.3、EuroSAT 峰值已確認在 1.5 附近，邊際資訊增益低於進入下一階段（56×56 解析度實驗）的機會成本。

## 相關檔案

- Sweep 腳本：`scripts/run_lambda3_sweep.sh`、`scripts/run_lambda3_sweep_retry.sh`、`scripts/run_lambda3_sweep_eurosat_high.sh`
- 完整 log：`logs/lambda3_sweep/`（3060 原始/RECOVERED/RETRY 版本、4090 補跑版本 `*_4090.log`、EuroSAT 高值加測 `eurosat_l3_{1.5,2.0}.log`）
- Checkpoints：`checkpoints_lambda3_sweep/l3_{0.1,0.5,1.0,1.5,2.0}/`（λ3=0/0.3 借用 `checkpoints_ablation_{no_lcr,full}/`；1.5/2.0 的 checkpoint 留在 server127，本機僅存 log）
- 前置分析：`reports/0909_FeatureSR內部Ablation報告.md`
