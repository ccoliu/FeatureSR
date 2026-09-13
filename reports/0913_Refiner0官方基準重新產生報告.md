# 用 Refiner=0 重新產生官方基準數字報告
最後更新：2026-09-13

## 背景

0909_FeatureSR內部Ablation報告.md 的建議 #3——「用 FeatureUpsampler-only + 依資料集決定的 $\mathcal{L}_{CR}$ 重新產生一組 5-shot / 1-shot 的正式基準結果」——一直沒有執行，導致論文主表（methodology 文件 §4.2-§4.7）用的是 `refiner=2`，跟論文自己推薦的最終架構（`refiner=0`）不一致。本報告記錄這次補跑與 `docs/FeatureSR_Methodology.md`／`README.md` 的同步更新（v2.4 → v2.5）。

## 執行範圍

四個 Phase，全部在 server127（RTX 4090）完成，`scripts/run_refiner0_official_regen.sh`：

| Phase | 內容 | 資料集 | 協定 |
|:---|:---|:---:|:---|
| 1 | FSR standalone 1-shot | 4 | 比照 `run_all_strict.sh`（lr=1e-4，無 warmup，100 episodes） |
| 2 | FSR+Ensemble 5-shot | 4 | 比照 `run_fsr_ensemble_pilot.sh`（lr=3e-5，warmup=5，400 episodes） |
| 3 | FSR+Ensemble 1-shot | 4 | 比照 `run_1shot_full.sh` 的 FSR+Ensemble 區塊 |
| 4 | TTA 重新驗證 | 4 | 用 Phase 2 的新 checkpoint，純評估 |

FSR standalone 5-shot（§4.3 核心基準）不必重跑，沿用既有乾淨數字：0909 ablation C 組（EuroSAT/CropDiseases/ISIC）+ 0912 56×56 report 的 control 組（ChestX）。Baseline、Ensemble-only（不含 FSR）不受 refiner 影響，未重跑。

20 個 log 全數正常完成，無崩潰。

## 結果總覽（refiner=2 舊值 → refiner=0 新值）

**FSR standalone 1-shot**：EuroSAT 75.31→77.77、CropDiseases 78.81→76.53、ISIC 33.40→33.85、ChestX 20.68→21.09（全部落在信賴區間內，refiner 對 1-shot FSR standalone 也無實質影響，延伸了 0909 report 只驗證過 5-shot 的結論）。

**FSR+Ensemble 5-shot**：EuroSAT 94.54→94.33、CropDiseases 90.97→90.11、ISIC 44.77→45.43、ChestX 22.96→22.98。CropDiseases 的變化值得注意：新數字（90.11%）已經低於 Ensemble-only（90.57%），使得「4/4 資料集嚴格單調遞增」的舊敘述不再成立（詳見下方修正說明）。

**FSR+Ensemble 1-shot**：CropDiseases 從 +2.86%（文件中原本最大的單一 Δ）縮水到 +0.54%，落入自身信賴區間，不再是穩健效果；四個資料集的平均 Δ 從 +0.67% 轉為 -0.31%。

**TTA（用新 checkpoint 重新驗證）**：ISIC 的退步從顯著的 -1.38%（超過信賴區間和 1.36%）縮小為不顯著的 -0.85%（信賴區間和 1.35%）——退步方向沒變，但已經無法排除是雜訊。

## 順手修正的兩個既有文件錯誤

在重寫這幾個章節時，發現兩處與本次改動無關、但確實是誤述的舊內容，一併修正：

1. **§4.5 舊稱「4/4 資料集嚴格單調遞增」**：即使在舊的 `refiner=2` 數字下，ChestX 的 Baseline（22.88%）本來就高於 Ensemble-only（22.79%），這個宣稱對 ChestX 從來就不成立，只是沒被抓出來。
2. **§4.6 舊稱 Ensemble-only ChestX 的 TTA +0.83% 是「對 ChestX 最有效的單一手段」**：套用文件自己訂的顯著性判準（Δ 超過雙邊信賴區間之和），+0.83% 並未超過 0.98% 的門檻，嚴格來說不算顯著效果，只是該資料集上觀察到的最大點估計值。

## 文件更新

`docs/FeatureSR_Methodology.md` 版本 v2.4 → v2.5：
- §4.3（1-shot + 5-shot FSR 表格）、§4.5（FSR+Ensemble 5-shot）、§4.6（TTA）、§4.7（FSR+Ensemble 1-shot）全部替換為 refiner=0 數字，並重寫對應分析文字
- §5 Computational Complexity：移除 FeatureRefiner 參數行，Total Added by FeatureSR 從 10.13M 改為 5.92M
- §6.2 重寫：ISIC TTA 退步從「確認的真實效果」改為「方向一致但不再顯著」，並明確指出 $\mathcal{L}_{CR}$ 單獨的貢獻仍是未跑過的開放問題
- 新增 §6.3：說明為何採用 refiner=0 作為正式架構，並交代 λ3 敏感度分析（用 refiner=2 跑的）為何不需要重跑
- 新增 v2.4→v2.5 changelog

`README.md` 同步更新表格 1、2、4、5、6（表格 3 Ensemble-only 不受影響，未動）。

## 尚未做的事（如實記錄）

- $\mathcal{L}_{CR}$ 對 ISIC TTA 退步的獨立貢獻仍未隔離驗證（§6.2 開放問題，前次盤點的「還沒做的實驗 #2」）——refiner=0 讓退步幅度減半，但沒有直接證據排除 $\mathcal{L}_{CR}$ 本身的作用。
- λ3 敏感度分析（0910 report）仍是用 `refiner=2` 跑的，未重新驗證於 `refiner=0` 下是否成立。§6.3 已說明為何預期結論不受影響、暫不重跑。
- `docs/FeatureSR_Methodology.tex`/`.pdf` 仍未同步 v2.5 的變動，依先前決議留到寫論文時一併補上。

## 相關檔案

- 重跑腳本：`scripts/run_refiner0_official_regen.sh`
- 完整 log：`logs/refiner0_regen/phase{1,2,3,4}_*/`
- Checkpoints：`checkpoints_refiner0/{fsr_1shot,fsr_ensemble_5shot,fsr_ensemble_1shot}/`（在 server127 上，本機僅存 log）
- 前置分析：`reports/0909_FeatureSR內部Ablation報告.md`、`reports/0912_56x56解析度實驗報告.md`
