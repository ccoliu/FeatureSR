# Prompt Ensemble 訓練/測試分布不一致問題：診斷與修復
# 最後更新：2026-08-29

## 背景

`utils/prompt_templates.py` 實作了領域自適應的 Multi-Prompt Ensemble（多模板文字特徵集成 + 類別名稱語意展開），原先設計為**推論時**的增強手段：訓練照舊用單一固定模板 `"a photo of a {class_name}"`，只在測試（`evaluate.py` / `experiments/evaluate_enhanced.py`）時才切換成 ensemble 版本的文字特徵。

初步測試（`logs/enhanced_5shot_real.log`）發現，這個 post-hoc（訓練/測試不一致）用法在已經用 LoRA + Cycle-Consistency 微調過的模型上**普遍造成負面效果**：

| 資料集 | Baseline（標準 prompt）| Post-hoc Ensemble | Delta |
|---|---:|---:|---:|
| EuroSAT | 90.17 ± 0.89% | 84.15% | **-6.03%** |
| CropDiseases | 90.03 ± 1.36% | 85.04% | **-4.99%** |
| ISIC | 43.76 ± 1.38% | 25.15% | **-18.61%** |
| ChestX | 22.88 ± 0.83% | 21.28% | **-1.60%** |
| 平均 | 61.71% | 53.90% | **-7.81%** |

值得注意：同一套 Prompt Ensemble 若套用在「checkpoint 未正確載入」（等同 zero-shot CLIP）的模型上，反而帶來 +16~19% 的巨大增益（`logs/enhanced_5shot_fsr.log`）。這個對比是關鍵線索。

## 根因分析

`models/clip_lora.py` 的 LoRA 只掛在 **CLIP ViT 視覺塔（visual encoder）的 attention projection** 上；文字塔（text encoder）完全凍結，不參與訓練。

訓練時（`train_episode`）文字特徵是在 `torch.no_grad()` 下算出來，作為 T-I-T / I-T-I 循環一致性損失與分類損失的**固定目標**，梯度只回傳到視覺端的 LoRA 參數。也就是說：**視覺端 LoRA 是針對「訓練時所用的那個固定 prompt 模板」去校準影像-文字對齊的**。

測試時如果臨時換成 Prompt Ensemble（多模板平均 + 類別名稱語意展開，例如把 ISIC 的 `"MEL"` 展開成整句 `"melanoma, a malignant pigmented skin cancer lesion"`），文字特徵的分布與訓練時完全不同，視覺端 LoRA 已經校準好的對齊關係直接被打亂。**展開幅度越大、掉分越嚴重**（ISIC 的語意展開幅度最大，掉分也最多），這個梯度完全符合觀察到的規律。

而 zero-shot CLIP（沒有 LoRA 微調）沒有這種「訓練時校準點」的問題，因此 Prompt Ensemble 對它是純粹的正面增益——這是 CLIP 圈子裡常見的 prompt engineering 效果，但不適用於已經微調過的模型。

## 修復方案

既然視覺端 LoRA 的訓練目標可以自由選擇文字特徵的生成方式，讓**訓練時就使用與測試時一致的 Prompt Ensemble**，使 LoRA 直接對齊到 ensemble 後的文字分布，而不是訓練用單一模板、測試才切換。

`train.py` 已預留 `--use_prompt_ensemble` 開關貫穿 `train_episode` 與 `evaluate`，但實際執行時發現一個獨立的程式 bug：`train_episode` 內部引用了不存在於該函式作用域的全域變數 `args`（`NameError: name 'args' is not defined`），導致這個路徑從未被真正跑過。修復方式：把 `dataset_name` 與 `use_prompt_ensemble` 改為 `train_episode` 的顯式參數，由呼叫端 `train_on_dataset` 傳入。

## 驗證實驗設定

- 4 個資料集、5-way 5-shot、Strict Few-Shot、100 epochs、`lr=3e-5` + `warmup_epochs=5`
- 各資料集 λ1/λ2 沿用論文 Table 13 配置（EuroSAT 1.5/0.2、CropDiseases 1.0/1.5、ISIC 3.0/2.0、ChestX 3.0/0.5）
- 訓練與最終評估皆加 `--use_prompt_ensemble`，checkpoint 存於獨立的 `checkpoints_ensemble/` 目錄（不覆蓋原始 baseline 權重）
- ChestX 因終端機視窗意外關閉，訓練在 epoch 80 的 eval 階段被 CUDA context 中斷；由於 epoch 30~70 的 eval 已經全部低於 epoch 20 的最佳值（過擬合下滑趨勢明確），改為直接用 epoch 20 checkpoint 執行一次獨立的 400-episode 最終評估（`evaluate.py --use_prompt_ensemble`），與其餘三個資料集方法論一致

## 結果

| 資料集 | Baseline（訓練/測試皆標準 prompt）| Post-hoc Ensemble（不一致）| **Ensemble-aware 訓練**（一致）| Best Epoch | 相對 Baseline |
|---|---:|---:|---:|:---:|---:|
| ISIC | 43.76 ± 1.38% | 25.15%（-18.61%）| **44.49 ± 0.61%** | 10 | **+0.73%** |
| EuroSAT | 90.17 ± 0.89% | 84.15%（-6.03%）| **93.81 ± 0.32%** | 30 | **+3.64%** |
| CropDiseases | 90.03 ± 1.36% | 85.04%（-4.99%）| **90.57 ± 0.64%** | 70 | **+0.54%** |
| ChestX | 22.88 ± 0.83% | 21.28%（-1.60%）| 22.79 ± 0.45% | 20 | -0.09%（誤差範圍內持平）|
| **平均** | **61.71%** | **53.90%（-7.81%）** | **62.92%** | — | **+1.21%** |

## 結論與後續

1. **假設完全成立**：訓練/測試文字分布不一致才是 Prompt Ensemble 崩潰的根因，將 ensemble 提前烘進 LoRA 訓練即可修復，且 3/4 資料集額外取得正向增益，整體平均 **+1.21%**，量級與 Feature-SR 本身的增益（+1.02%，見 `reports/0826成果報告.md`）相當，兩者是獨立的增益來源，理論上可疊加。
2. **ChestX 是例外**：完全修復了崩潰（-1.60% → -0.09%），但沒有額外增益，這與 Feature-SR 在 ChestX 上增益也最小（+0.03%）的模式一致——胸部 X 光這個領域對這類文字端/特徵端的增強手段本來就不敏感。
3. **收斂/過擬合速度明顯變快**：baseline 通常要練到 epoch 30~90 才見頂，ensemble-aware 訓練的 best epoch 分別落在 10（ISIC）、20（ChestX）、30（EuroSAT）、70（CropDiseases），落差很大且普遍早於對應 baseline。目前 `--eval_interval 10` 的粒度可能會錯過真正峰值（尤其 ISIC 在 epoch 10 就已經是峰值，epoch 5~9 之間未知），後續調參建議把 eval_interval 縮小或改用 early stopping。
4. **待驗證**：1-shot 設定、與 Feature-SR（28×28）疊加使用的效果。

## Checkpoints 位置

`checkpoints_ensemble/`：
- `isic_5shot_best.pth`（epoch 10, 44.49%）
- `eurosat_5shot_best.pth`（epoch 30, 93.81%）
- `crop_disease_5shot_best.pth`（epoch 70, 90.57%）
- `chestx_5shot_best.pth`（epoch 20, 22.79%，最終 100 epoch 未跑完，見上述說明）

## 相關檔案

- 根因與修復：`train.py`（`train_episode` 函式簽名新增 `dataset_name` / `use_prompt_ensemble` 參數）
- Prompt Ensemble 實作：`utils/prompt_templates.py`
- 訓練 log：`logs/ensemble_pilot/{dataset}_5shot_ensemble.log`
- ChestX 補評估 log：`logs/ensemble_pilot/chestx_5shot_final_eval.log`
- 批次執行腳本：`scripts/run_ensemble_pilot.sh`
