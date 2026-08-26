# CC-CDFSL 復現實作計劃

## 論文摘要

**論文**：Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment (CVPR 2026)  
**arXiv**：2603.17655  
**GitHub**：https://github.com/z-yaz/CC-CDFSL（目前尚無程式碼，僅有海報）

## 核心問題

CLIP 在跨域少樣本學習（CDFSL）中，Domain Gap 和 scarce training data 會使 **局部 patch 特徵**與文字語義的對齊比全域 CLS 特徵更嚴重地退化，即「local misalignment problem」。

## 方法（CC-CDFSL）

### Preliminaries
- **設定**：K-way N-shot（論文中使用 5-way 1-shot / 5-shot）
- **Backbone**：CLIP ViT-Base/16（主要）
- **PEFT 方法**：可搭配 CoOp, CLIP-Adapter, MaPLe, CLIP-LoRA 等

### 關鍵符號
- `T`: 文字特徵矩陣 `[C, d]`（C = 類別數，d = 特徵維度）
- `L`: 所有圖像的 patch 局部特徵 `[B*(A+1)*P, d]`（B = batch，A = augment 次數，P = patch 數）
- `D_txt`: 文字特徵與 patch 特徵的相似度矩陣

### Semantic Anchor Module - Augmentation Phase

對每張圖像進行 A 次 augmentation（隨機水平翻轉、旋轉等），擴大 patch 特徵語料庫：

```
X_aug ∈ R^{B*A*P × d}   (augmented patch features)
L ∈ R^{B*(A+1)*P × d}   (original + augmented patches)
```

### Text-to-Image-to-Text (T-I-T) Cycle

1. 計算文字-patch 相似度矩陣（Eq. 4-6）：
   ```
   D_txt[j, i] = T_j · L_i      (cosine similarity)
   ```

2. 對每個文字特徵選最相似 patch（Eq. 7）：
   ```
   L*_j = L[argmax_i D_txt[j, i]]
   ```

3. 計算反向相似度（Eq. 8）：
   ```
   E_txt = L* · T^T ∈ R^{C×C}
   ```

4. 選最相似的文字重建（Eq. 9）：
   ```
   T^rec_j = T[argmax_k E_txt[j, k]]
   L_cyc_txt = 1 - (1/C) * sum_j sim(T_j, T^rec_j)
   ```

### Semantic Anchor Module - Shrinking Phase

針對 I-T-I cycle，先過濾無關的 patch：

1. 對每個影像 b 和每個類別 j，選 top-k 相似 patches（Eq. 10）：
   ```
   I^(b)_j = top-k_i(D_txt[b, j, i])
   ```

2. 合併去重（Eq. 11）：
   ```
   I_anchor = unique(∪_b ∪_j I^(b)_j),  V = |I_anchor|
   X_anchor = L[I_anchor] ∈ R^{V×d}
   ```

### Image-to-Text-to-Image (I-T-I) Cycle

1. 對每個 anchor patch，找最相似文字（Eq. 13）：
   ```
   t_n = T[argmax_j (x_n · T_j)]
   ```

2. 用文字作橋接，在 augmented space 中找最相似 patch（Eq. 14）：
   ```
   m* = argmax_m(t_n · X_aug[m])
   x_hat_n = X_aug[m*]
   ```

3. I-T-I 損失（Eq. 15）：
   ```
   L_cyc_img = 1 - (1/V) * sum_n sim(x_n, x_hat_n)
   ```

### 總損失（Eq. 16）
```
L_total = L_CE + λ1 * L_cyc_txt + λ2 * L_cyc_img
```

## 超參數設定（論文）

| 參數 | 值 |
|------|-----|
| backbone | CLIP ViT-B/16 |
| epochs | 100 |
| k (Semantic Anchor top-k) | 10 |
| A (augmentation 次數) | 需從論文附錄確認，預設 4 |
| λ1, λ2 | 透過 grid search 設定（附錄詳述） |
| GPU | NVIDIA RTX 4090 |

## 資料集

| 資料集 | 類別 | 內容 |
|--------|------|------|
| CropDiseases | 38 | 植物病害 |
| EuroSAT | 10 | 衛星影像 |
| ISIC2018 | 7 | 皮膚病變 |
| ChestX | 7 | 胸部X光 |

## 開放問題（需使用者確認）

> [!IMPORTANT]
> **問題 1**：復現哪個 PEFT 基底方法？
> - CLIP-LoRA（論文中表現最好，推薦）
> - CLIP-Adapter
> - CoOp
> - MaPLe
> - 全部？

> [!IMPORTANT]
> **問題 2**：您有這些資料集的下載路徑嗎？
> - 如果沒有，需要提供自動下載腳本

> [!NOTE]
> **問題 3**：λ1, λ2 超參數論文附錄提到用 grid search 決定，我將根據論文實驗設定 λ1=1.0, λ2=0.5 作為預設值，請確認是否接受

## 計劃程式碼結構

```
cc-cdfsl/
├── README.md
├── requirements.txt
├── train.py                     # 訓練主程式
├── evaluate.py                  # 評估主程式
├── configs/
│   └── default.yaml             # 預設超參數配置
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py          # 基底資料集類
│   ├── crop_disease.py          # CropDiseases
│   ├── eurosat.py               # EuroSAT
│   ├── isic.py                  # ISIC2018
│   └── chestx.py                # ChestX
├── models/
│   ├── __init__.py
│   ├── clip_wrapper.py          # CLIP 包裝，提取 patch features
│   ├── clip_lora.py             # CLIP-LoRA 適配
│   └── cc_cdfsl.py              # CC-CDFSL 核心模組
├── losses/
│   ├── __init__.py
│   └── cycle_consistency.py     # T-I-T 和 I-T-I 循環一致性損失
└── utils/
    ├── __init__.py
    ├── few_shot_sampler.py      # Few-shot task sampler
    ├── augmentation.py          # 影像增強
    └── metrics.py               # 評估指標
```

## 驗證計劃

- 使用 CLIP-LoRA + CC-CDFSL 在 EuroSAT 5-way 5-shot 測試
- 預期結果（論文 Table 1）：86.07% (vs. baseline 81.49%)
