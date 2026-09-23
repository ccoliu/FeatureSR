# ISIC TTA 退步的 λ3 隔離驗證：假設被推翻
最後更新：2026-09-13

## 背景

`docs/FeatureSR_Methodology.md` §6.2 長期掛著一個未驗證的假設：ISIC 2018 在 Feature-SR+Ensemble 疊加 TTA 時的退步，可能是 $\mathcal{L}_{CR}$（Cross-Resolution Consistency Loss）把視覺嵌入錨定得太緊、導致翻轉增強反而干擾決策邊界所致。`reports/0913_Refiner0官方基準重新產生報告.md` 完成後，這個假設仍是唯一沒有直接驗證過的開放問題。

## 方法

比照 `scripts/run_refiner0_official_regen.sh` Phase 2 的 ISIC 5-shot FSR+Ensemble 協定（refiner=0，5-way 5-shot，`lr=3e-5`，warmup=5，400 episodes），**只把 `lambda3` 從論文預設值 0.3 改成 0**，其餘完全不變，重新訓練一組，再對同一份 checkpoint 做 TTA 有/無對照。腳本：`scripts/run_isic_lambda3_0_tta_check.sh`。

若假設成立，預期 λ3=0 時 TTA 的退步應該消失或明顯縮小；若退步依然存在甚至惡化，代表原因不在 $\mathcal{L}_{CR}$。

## 結果

| 配置 | 無 TTA | +TTA | Δ | 信賴區間和 | 判定 |
|:---|:---:|:---:|:---:|:---:|:---:|
| refiner=2, λ3=0.3（舊，`reports`前次數字） | 44.13 ± 0.67% | 42.75 ± 0.69% | -1.38% | 1.36% | ✅ 顯著退步 |
| refiner=0, λ3=0.3（0913 report） | 45.48 ± 0.73% | 44.63 ± 0.62% | -0.85% | 1.35% | 不顯著 |
| **refiner=0, λ3=0（本報告）** | **45.54 ± 0.64%** | **43.80 ± 0.66%** | **-1.74%** | **1.30%** | **✅ 顯著退步，且是三者中最大的** |

## 分析

**假設被推翻**：如果 $\mathcal{L}_{CR}$ 是退步的成因，拿掉它（λ3=0）應該讓退步消失或縮小；實際觀察到的是退步不僅沒消失，幅度還從 refiner=0/λ3=0.3 的 -0.85%（不顯著）擴大回 -1.74%（顯著，且是三個配置中最大的）。$\mathcal{L}_{CR}$ 不是這個交互作用的成因——把它拿掉反而讓問題更明顯，說明它原本可能還有一點點抵消退步的作用（雖然 refiner=0/λ3=0.3 那組的 -0.85% 本身也不顯著，無法據此做強論斷）。

真正的成因更可能是以下兩者之一，但都超出本次驗證範圍：
1. **ISIC 資料域本身對水平翻轉的視覺特性敏感**（皮膚病灶影像的診斷線索可能帶有方向性資訊，例如病灶邊界的不對稱性判讀），與 Feature-SR 的任何元件都無關——這與 §6.2 原本「翻轉在語意上對皮膚鏡影像應該是合理的」的前提假設本身可能就有問題。
2. Ensemble-aware 訓練與 TTA 的交互作用，與 Feature-SR 無關（因為 Ensemble-only 配置在原始 `reports/0901_TTA驗證報告.md` 中對 ISIC 的 TTA 效果是 -0.07%，近乎中性，遠小於 FSR+Ensemble 疊加後的退步——代表退步需要 Feature-SR 存在才會被放大，但成因不是 $\mathcal{L}_{CR}$ 這個特定元件）。

## 結論與建議

1. **§6.2 的 $\mathcal{L}_{CR}$ 假設應該正式撤回**，改為誠實記錄「退步的成因不明，已排除 $\mathcal{L}_{CR}$」。
2. 不建議再花算力追查真正成因（例如逐一測試 FeatureUpsampler 本身、或資料集特性），因為：(a) 這是一個 §6 Known Limitations 等級的邊界陳述，不是核心 contribution；(b) 已經給出一個更保守但誠實的結論：TTA 對 ISIC + Feature-SR 的效果不穩定且方向持續為負，直接建議關閉即可，不需要精確定位成因才能做出這個工程決策。
3. `docs/FeatureSR_Methodology.md` §6.2 已同步更新（見下方文件變更）。

## 文件變更

`docs/FeatureSR_Methodology.md` §6.2 重寫：撤回 $\mathcal{L}_{CR}$ 假設，改為記錄三組配置（refiner=2/λ3=0.3、refiner=0/λ3=0.3、refiner=0/λ3=0）的完整對照，並明確聲明「已排除 $\mathcal{L}_{CR}$ 是成因」。版本 v2.5 → v2.6。

## 相關檔案

- 驗證腳本：`scripts/run_isic_lambda3_0_tta_check.sh`
- 完整 log：`logs/isic_lambda3_0_tta_check/`
- 前置分析：`reports/0913_Refiner0官方基準重新產生報告.md`、`reports/0901_TTA驗證報告.md`
