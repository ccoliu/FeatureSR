# Related Work: Feature Upsampling / Feature Super-Resolution
# 定位分析與 baseline 選擇建議
# 最後更新：2026-09-06

本文件整理 feature upsampling 這條研究線的既有工作，釐清 FeatureSR 的 novelty 邊界應該畫在哪裡，以及哪一篇最適合當實驗 baseline。所有技術細節皆取自各論文原文（摘要或全文），非二手轉述。

---

## 1. 既有工作對照表

| 工作 | 「upsampling」在該工作的意義 | 機制 | 訓練目標 / consistency | 需要 image guidance？ | 推論時是否運作 | 下游任務 |
|:---|:---|:---|:---|:---:|:---:|:---|
| **FSR-GAN** [Tan+, CVPR 2018] | **維度不變的特徵「還原」**：把小圖產生的劣化特徵映射成「若原圖是大圖時該有的特徵」。不改變空間網格大小 | Feature-space GAN（generator G + discriminator D 交替訓練） | 對抗式；以大圖的真實特徵作為 ground truth | 否 | 是（強化後的 query 特徵直接用於檢索） | 影像檢索（Oxford5K / Paris / Holidays / Flickr100k） |
| **FeatUp** [ICLR 2024] | 空間解析度提升，產出任意解析度的 dense features | 兩種變體：(a) JBU 式 guided 單次前向；(b) 對單張影像擬合 implicit model | **Multi-view reconstruction loss**：對輸入做 pad/zoom/crop/flip 等 jitter，要求「預測的高解析特徵經同樣 jitter 後再 downsample」能重建出 backbone 在該 jitter 視角下的低解析特徵。使用習得的 downsampler（blur kernel 或 attention-adaptive）+ 不確定度加權 L2 | 是（變體 a 需高解析影像訊號引導） | 是（高解析特徵被下游消費） | CAM、transfer 分割/深度、end-to-end 語意分割 |
| **UPLiFT** [CVPR 2026] | 空間解析度提升至 pixel-dense | **Local Attender**：完全局部的 attentional pooling，在固定鄰域內以習得權重聚合，避免全域 QKV attention；迭代式上採樣 | 論文摘要未載明；強調保持 backbone 特徵分佈 | 未載明 | 是（主打低推論成本） | 語意分割、單目深度、影像 SR、text-to-image 生成 |
| **ViT-Up** [arXiv 2606.14024, 2026-06] | ViT patch token 網格解析度提升 | **Implicit upsampling**：從 ViT 各層 hidden states 建構 query，可在任意連續影像座標預測特徵 | 論文摘要未載明 | **否**（明確取代外部影像引導，避免 feature leakage / fragmentation / blur） | 是 | 語意分割（Cityscapes）、深度估計、語意對應（SPair-71k） |
| **本工作 FeatureSR** | ViT patch 網格 14×14 → 28×28 | Bilinear 插值 + 殘差卷積校正 + 可學習位置編碼 + 2 層 Pre-LN Transformer refiner | $\mathcal{L}_{CR}$：AdaptiveAvgPool 回 14×14 後與 $F_{orig}$ 取 cosine similarity；**外加** T-I-T / I-T-I 跨模態循環一致性（λ1, λ2）與 CE | 否 | **否 —— 推論時完全不參與** | 跨域少樣本分類（BSCD-FSL 四資料集） |

---

## 2. 關鍵發現一：$\mathcal{L}_{CR}$ 在結構上是 FeatUp reconstruction loss 的特例

FeatUp 的重建目標（原文式）：

$$
\mathcal{L}_{rec} = \frac{1}{|T|}\sum_{t \in T} \frac{1}{2s^2}\left\| f(t(x)) - \sigma_\downarrow(t(F_{hr})) \right\|_2^2 + \log(s)
$$

本工作的 $\mathcal{L}_{CR}$：

$$
\mathcal{L}_{CR} = 1 - \overline{\cos\left(\text{AdaptiveAvgPool}(F_{sr}),\; F_{orig}\right)}
$$

兩者是**同一個形狀**：把預測的高解析特徵降採樣回原解析度，與 backbone 實際輸出的低解析特徵比對。差異在於：

| 面向 | FeatUp | 本工作 |
|:---|:---|:---|
| 視角數 | 多視角（$T$ = pad / zoom / crop / flip 及其組合） | **單一視角**（僅恆等變換） |
| Downsampler | 習得的（blur kernel 或 attention-adaptive） | 固定的 AdaptiveAvgPool |
| 距離 | 不確定度加權 L2（轉成 proper likelihood） | Cosine similarity |
| 在整體目標中的角色 | **就是**訓練 upsampler 的全部訊號 | 四項損失中權重 $\lambda_3=0.3$ 的一項；upsampler 同時被 T-I-T/I-T-I 塑造 |

**結論**：不能宣稱「首次提出 consistency-based feature upsampling」，也不能宣稱「pool 回去比對」這個機制本身是新的。

**且有一個實質的技術推論**：FeatUp 的多視角 jitter 正是它能夠恢復 **sub-token 級細節**的來源——不同偏移的視角提供了單一視角下不存在的約束（概念上近似 tomography / NeRF 的多視角重建）。本工作只用單一視角，因此 $\mathcal{L}_{CR}$ **在資訊論上無法注入任何新的空間細節**，它只能作為「防止升採樣特徵漂離預訓練語意空間」的錨定正則項。這與 §4.2 觀察到的 Feature-SR 增益幅度偏小（5-shot +1.02%）是相符的，也讓「這是不是只是 bilinear 插值 + 額外參數量」這個質疑更需要正面回應。

**直接影響的實驗優先序**：`--lambda3 0`（關掉 $\mathcal{L}_{CR}$）與 `--sr_refiner_layers 0`（關掉 refiner）這兩個 ablation 從「應該做」升級為「**必須做**」。若 $\lambda_3=0$ 幾乎不掉分，則 $\mathcal{L}_{CR}$ 這個賣點不成立；若關掉 refiner 才是掉分主因，則增益來源是額外容量而非「超解析」。

---

## 3. 關鍵發現二：真正可防守的差異在「模組的角色」，不在「升採樣」

上表最後兩欄揭示了一個所有既有工作都不具備的性質：

> **FeatUp、UPLiFT、ViT-Up、FSR-GAN 全部都是 feature producer——升採樣後的特徵是產品，會被下游任務在推論時消費。**
>
> **本工作的升採樣網格在推論時完全不參與運算。**

具體證據（程式碼）：`train.py:250-259` 中，$F_{sr}$ 僅餵入 T-I-T / I-T-I 的 argmax 檢索網格與 $\mathcal{L}_{CR}$；query 分類走的是 backbone 的 CLS token，而 `evaluate()`（`train.py:312-352`）根本沒有 `feature_sr` 參數。因此 FeatureSR 的增益 **100% 透過「在訓練期塑造 LoRA 權重」實現**，模組本身在部署時可以整個丟棄。

這是一個與既有文獻正交的定位：**把 feature upsampling 當作訓練期的鷹架（scaffold），用來提供更細粒度的跨模態檢索網格，而非當作要輸出的表徵**。

**但這個定位有一個必須誠實面對的後果**：既然沒有產出任何高解析表徵給任何人使用，「Super-Resolution」這個命名的正當性比原先設想的更薄弱。命名的修正方向或許不是換一個 upsampling 的同義詞，而是往「描述實際作用」的方向走（例如強調 fine-grained cross-modal supervision / training-time scaffold）。

---

## 4. 命名衝突評估

[Tan+, CVPR 2018] 已經佔用了 **Feature Super-Resolution** 這個術語。需要注意的是：**該工作與本工作的機制完全不同**——FSR-GAN 是「維度不變的特徵還原」，有明確的 ground truth（大圖的真實特徵），解決的是小圖檢索問題；本工作是「空間網格密化」，額外的 token 沒有任何 ground truth。

因此這是一個**純粹的命名衝突，而非技術撞題**，但 reviewer 仍會第一時間 pattern-match 到該篇。建議在 related work 中主動點名並清楚切割，並認真考慮更名。

---

## 5. Baseline 選擇建議

**建議以 FeatUp 為主要對照組**，理由：

1. 它是唯一在**訓練目標結構上與 $\mathcal{L}_{CR}$ 重疊**的工作，直接回答「你的 consistency 有什麼特別」這個必然會被問的問題。
2. 它是 model-agnostic 的，且已在 ViT 系 backbone 上驗證，接入 CLIP ViT-B/16 的阻力最低。
3. ViT-Up 的評估集中在 DINOv3 且發表僅數月；UPLiFT 主打 pixel-dense 與生成任務，兩者與本工作的設定距離較遠，適合列入 related work 討論但不必然要跑實驗。

**實作注意**：因 §3 所述的架構差異，這**不是 drop-in 替換**。不能直接「換成 FeatUp 然後比分類準確率」——必須把 FeatureUpsampler + FeatureRefiner 換成 FeatUp，作為**餵給 cycle-consistency 損失的網格產生器**（即替換 `apply_feature_sr` 的內容），其餘訓練流程不變。

這樣定義出的實驗問題是乾淨的：

> 為 dense prediction 設計的通用 feature upsampler，能否勝任 CDFSL 中的跨模態檢索鷹架角色？若不能，為什麼 CDFSL 需要任務特化的升採樣機制？

---

## 6. 參考連結

- FSR-GAN (CVPR 2018): https://openaccess.thecvf.com/content_cvpr_2018/html/Tan_Feature_Super-Resolution_Make_CVPR_2018_paper.html
- FeatUp (ICLR 2024): https://arxiv.org/abs/2403.10516
- UPLiFT (CVPR 2026): https://arxiv.org/abs/2601.17950 ・ code: https://github.com/mwalmer-umd/UPLiFT
- ViT-Up (arXiv 2606.14024, 2026-06): https://arxiv.org/abs/2606.14024
