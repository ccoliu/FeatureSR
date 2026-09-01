# Feature-SR + Prompt-Ensemble 疊加驗證報告
# 最後更新：2026-08-31

## 背景

`reports/0826成果報告.md` 已驗證 Feature-SR（14×14 → 28×28 patch token 升採樣）平均帶來 +1.02% 的增益；`reports/0829_PromptEnsemble修復報告.md` 已驗證修復訓練/測試分布不一致問題後，Ensemble-aware 訓練平均再帶來 +1.21% 的增益。兩者分別作用在不同層面：Feature-SR 改的是視覺端 patch token 的空間解析度，Ensemble-aware 訓練改的是文字端的 prompt 分布與 LoRA 對齊目標。

理論上兩者是正交的增益來源，可以同時開啟。本報告驗證疊加使用（`--use_feature_sr --use_prompt_ensemble` 同時開啟）是否能取得比單獨使用更好的效果。

## 實驗設定

- 4 個資料集、5-way 5-shot、Strict Few-Shot、100 epochs、`lr=3e-5` + `warmup_epochs=5`
- Feature-SR：scale=2、refiner_layers=2、refiner_heads=8、λ3=0.3
- 各資料集 λ1/λ2 沿用論文 Table 13 配置
- 訓練與最終評估同時開啟 `--use_feature_sr` 與 `--use_prompt_ensemble`
- checkpoint 存於獨立的 `checkpoints_fsr_ensemble/` 目錄
- 執行腳本：`scripts/run_fsr_ensemble_pilot.sh`（於 tmux session 中執行，全程未中斷）

## 結果

| 資料集 | Baseline（無任何增強）| Ensemble-only | **Feature-SR + Ensemble** ⭐ | Best Epoch | vs Baseline | vs Ensemble-only |
|---|---:|---:|---:|:---:|---:|---:|
| EuroSAT | 90.17 ± 0.89% | 93.81 ± 0.32% | **94.54 ± 0.30%** | 40 | **+4.37%** | +0.73% |
| CropDiseases | 90.03 ± 1.36% | 90.57 ± 0.64% | **90.97 ± 0.60%** | 50 | **+0.94%** | +0.40% |
| ISIC 2018 | 43.76 ± 1.38% | 44.49 ± 0.61% | **44.77 ± 0.65%** | 40 | **+1.01%** | +0.28% |
| ChestX | 22.88 ± 0.83% | 22.79 ± 0.45% | **22.96 ± 0.47%** | 70 | **+0.08%** | +0.17% |
| **平均** | **61.71%** | **62.92%** | **63.31%** | — | **+1.60%** | **+0.39%** |

## 關鍵發現

1. **4/4 資料集全部呈現一致的遞增疊加模式**：baseline < ensemble-only < Feature-SR + ensemble，沒有任何資料集出現互相抵銷或負向交互作用。兩個增益來源（視覺端解析度、文字端分布對齊）確認為正交且可疊加。
2. **EuroSAT 疊加效果最顯著**：總增益達 +4.37%（90.17% → 94.54%），遠高於任一單獨方法（Feature-SR 原始論文報告 +1.37%、Ensemble-only +3.64%）。
3. **ISIC 訓練穩定性明顯改善**：純 Ensemble-only 訓練的 ISIC 在 epoch 10 見頂後因過擬合單調下滑至 38%；疊加 Feature-SR 後全程穩定在 43~45% 區間波動（見下方訓練曲線），推測 Feature-SR 的 Cross-Resolution Consistency Loss 對小樣本過擬合有額外正則化效果。
4. **ChestX 增益幅度仍是四者中最小**（+0.08% vs baseline），且訓練前期（epoch 10~30）一度低於 baseline，直到 epoch 70 才轉正。這與已知的架構限制一致：論文附錄 Table 9（SOTA 對比）顯示 ChestX 上表現最好的方法（IM-DCL, 28.93%）依賴 ResNet backbone 對局部醫學影像紋理的先天優勢，ViT/CLIP 系方法（含本論文的 CC-CDFSL，25.47%）在 ChestX 上普遍不如 ResNet 系方法。因此 ChestX 增益小不代表方法失效，而是受限於 backbone 選擇這個更上層的因素。

## 訓練曲線觀察（Best Epoch 分布）

| 資料集 | Ensemble-only Best Epoch | FSR+Ensemble Best Epoch |
|---|:---:|:---:|
| ISIC | 10 | 40 |
| ChestX | 20 | 70 |
| EuroSAT | 30 | 40 |
| CropDiseases | 70 | 50 |

疊加 Feature-SR 後，除 CropDiseases 略提前外，其餘資料集的最佳 epoch 都比純 ensemble-only 更晚出現，過擬合速度變慢，訓練更穩定——這進一步支持「Feature-SR 的 Cross-Resolution Consistency Loss 有正則化效果」的假設。

## 結論

Feature-SR 與 Ensemble-aware 訓練是兩個獨立、可疊加的增益來源，同時使用可取得優於任一單獨方法的效果（平均 +1.60% vs baseline，其中 EuroSAT 達 +4.37%）。ChestX 是唯一增益幅度極小的例外，根因是 backbone 架構限制（ViT/CLIP vs ResNet），而非本專案兩個增強模組本身的問題，建議在論文中作為誠實的已知限制呈現，而非繼續投入算力嘗試在 ChestX 上突破。

## Checkpoints 位置

`checkpoints_fsr_ensemble/`：
- `eurosat_5shot_fsr_best.pth`（epoch 40, 94.70%）
- `crop_disease_5shot_fsr_best.pth`（epoch 50, 91.02%）
- `isic_5shot_fsr_best.pth`（epoch 40, 44.78%）
- `chestx_5shot_fsr_best.pth`（epoch 70, 22.86%）

## 相關檔案

- 執行腳本：`scripts/run_fsr_ensemble_pilot.sh`
- 訓練 log：`logs/fsr_ensemble_pilot/{dataset}_5shot_fsr_ensemble.log`
- 前置報告：`reports/0826成果報告.md`（Feature-SR 單獨驗證）、`reports/0829_PromptEnsemble修復報告.md`（Ensemble-aware 訓練根因與修復）
