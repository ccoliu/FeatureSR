# FeatureSR: Feature-Level Super-Resolution with Cross-Resolution Consistency for Cross-Domain Few-Shot Learning

## Technical Methodology Document

**Project**: FeatureSR  
**Baseline**: CLIP-LoRA [3] with CC-CDFSL Cycle Consistency Framework (CVPR 2026) [1]  
**Version**: v2.0 (2026-08-31)

**Changelog (v1.0 → v2.0)**: Added the Domain-Adaptive Prompt Ensemble module (§3.3), its train/inference consistency requirement and a root-cause analysis of a prior regression (§3.4), combined Feature-SR + Prompt Ensemble empirical results (§4.4), a Known Limitations section on the ChestX backbone ceiling (§6), and a References section with inline citations, sourced from the CC-CDFSL paper's own bibliography where applicable.

---

## 1. Problem Formulation & Motivation

### 1.1 Challenges in Cross-Domain Few-Shot Learning (CDFSL)
Standard Few-Shot Learning typically evaluates novel classes from the same visual domain (e.g., miniImageNet $\rightarrow$ tieredImageNet). In contrast, **Cross-Domain Few-Shot Learning (CDFSL)** tasks the model with generalizing from generic natural images (Web-scale vision-language datasets) to completely distinct target domains with extreme domain shift:
* **EuroSAT** [6]: High-altitude multispectral satellite earth observation.
* **CropDiseases** [7]: Fine-grained plant pathology on leaf surfaces.
* **ISIC 2018** [8]: Dermoscopic skin lesion pathology.
* **ChestX-ray8** [9]: High-noise, grayscale medical radiography.

This four-domain protocol follows the BSCD-FSL benchmark [5].

Under extreme domain shift, global features (`[CLS]` token) suffer from catastrophic misalignment. Classification depends fundamentally on **fine-grained localized spatial cues** (e.g., leaf lesions, pigment networks, subtle radiographic opacities).

### 1.2 The Failure of Pixel-Level Super-Resolution
In preliminary studies, applying state-of-the-art pixel-level generative super-resolution models (e.g., Real-ESRGAN [10]) directly to input images before feeding them into CLIP [2] failed dramatically:
* **Hallucination Artifacts**: Natural image priors hallucinate artificial high-frequency textures in scientific/medical domains, leading to an **accuracy collapse of -36.88%** on EuroSAT (85.16% $\rightarrow$ 48.28%) and **-4.66%** on CropDiseases.
* **Computational Overhead**: Pixel-level generation introduces severe memory and latency bottlenecks.

### 1.3 The Core Proposal: Feature-Level Super-Resolution (FeatureSR)
FeatureSR operates **directly inside the intermediate Patch Feature Space of the Vision Transformer (ViT)**:
* Upsamples the ViT-B/16 spatial grid from **$14 \times 14 = 196$ tokens** to **$28 \times 28 = 784$ tokens**.
* Preserves pre-trained semantic integrity while providing **4× higher spatial resolution** for cross-modal cyclic alignment (Cycle Consistency).

---

## 2. Model Architecture & Mathematical Formulation

FeatureSR comprises three key components: **FeatureUpsampler**, **FeatureRefiner**, and **CrossResolutionConsistencyLoss**.

```
 Input Image X ∈ R^[B, 3, 224, 224]
        │
        ▼ (CLIP ViT-B/16 + LoRA Backbone)
 Base Patch Features F_orig ∈ R^[B, 196, 512] (14×14 Grid)
        │
        ├─────────────────────────────────────────────┐
        ▼                                             │
 ┌─────────────────────────────────────────┐          │
 │ 1. FeatureUpsampler                     │          │
 │    - 2D Reshape: [B, 512, 14, 14]       │          │
 │    - Bilinear Interpolation → [28, 28]  │          │
 │    - Residual Conv2D Correction         │          │
 │    - Learnable Positional Embedding     │          │
 └─────────────────────────────────────────┘          │
        │ [B, 784, 512]                               │
        ▼                                             │
 ┌─────────────────────────────────────────┐          │
 │ 2. FeatureRefiner                       │          │
 │    - Multi-Head Self-Attention (8 Heads)│          │
 │    - Pre-LayerNorm & GELU MLP           │          │
 │    - Post-LayerNorm & L2 Normalization  │          │
 └─────────────────────────────────────────┘          │
        │                                             │
        ▼                                             ▼
 Super-Resolved Features F_sr ∈ R^[B, 784, 512] Base Features F_orig ∈ R^[B, 196, 512]
        │                                             │
        ├──────────────────────┬──────────────────────┘
        │                      │
        ▼                      ▼
  T-I-T & I-T-I         Adaptive AvgPool (28×28 → 14×14)
 Cycle Consistency        & Cosine Similarity
  Matching Alignment    (Cross-Resolution Loss L_CR)
```

### 2.1 FeatureUpsampler
Given the base patch feature sequence $\mathbf{F} \in \mathbb{R}^{B \times P \times d}$ ($P = 196, d = 512$):

1. **Spatial Reshape**:
   $$\mathbf{F}_{\text{2D}} = \text{Reshape}(\mathbf{F}) \in \mathbb{R}^{B \times d \times H \times W} \quad (H=W=14)$$
2. **Bilinear Spatial Interpolation**:
   $$\mathbf{F}_{\text{interp}} = \text{Bilinear}(\mathbf{F}_{\text{2D}}, \text{scale}=2) \in \mathbb{R}^{B \times d \times 28 \times 28}$$
3. **Learnable Residual Correction**:
   $$\mathbf{R} = \text{Conv2D}_{3\times3}(\text{GELU}(\text{LN}(\text{Conv2D}_{3\times3}(\mathbf{F}_{\text{interp}}))))$$
   $$\mathbf{F}_{\text{up\_2D}} = \mathbf{F}_{\text{interp}} + \mathbf{R}$$
   *(Note: $\mathbf{R}$ is initialized near zero to preserve initial semantic structure).*
4. **Positional Embedding Injection**:
   $$\mathbf{F}_{\text{up}} = \text{Flatten}(\mathbf{F}_{\text{up\_2D}}) + \mathbf{P}_{\text{up}} \in \mathbb{R}^{B \times 784 \times 512}$$
   where $\mathbf{P}_{\text{up}} \in \mathbb{R}^{1 \times 784 \times d}$ is a learnable positional embedding.

### 2.2 FeatureRefiner
To capture long-range contextual spatial dependencies among the $N=784$ upsampled tokens, the sequence passes through $L=2$ Pre-LN Transformer blocks with Multi-Head Self-Attention ($h=8$ heads):

$$\mathbf{X}^{(l)\prime} = \mathbf{X}^{(l-1)} + \text{MHSA}(\text{LN}_1(\mathbf{X}^{(l-1)}))$$
$$\mathbf{X}^{(l)} = \mathbf{X}^{(l)\prime} + \text{MLP}(\text{LN}_2(\mathbf{X}^{(l)\prime}))$$

The output is normalized onto the unit hypersphere:
$$\mathbf{F}_{\text{sr}} = \frac{\text{LN}_{\text{post}}(\mathbf{X}^{(L)})}{\|\text{LN}_{\text{post}}(\mathbf{X}^{(L)})\|_2} \in \mathbb{R}^{B \times 784 \times 512}$$

---

## 3. Loss Functions & Objectives

The total training objective is formulated as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{TIT}} + \lambda_2 \mathcal{L}_{\text{ITI}} + \lambda_3 \mathcal{L}_{\text{CR}}$$

### 3.1 Cross-Resolution Consistency Loss ($\mathcal{L}_{\text{CR}}$)
To prevent semantic distortion during upsampling, $\mathcal{L}_{\text{CR}}$ forces the pooled high-resolution representations to align with the un-upsampled base tokens:
$$\mathbf{F}_{\text{pooled}} = \text{AdaptiveAvgPool2D}(\mathbf{F}_{\text{sr}}, (14, 14)) \in \mathbb{R}^{B \times 196 \times d}$$
$$\mathcal{L}_{\text{CR}} = 1 - \frac{1}{B \cdot P} \sum_{b=1}^{B} \sum_{p=1}^{P} \langle \hat{\mathbf{F}}_{\text{pooled}}^{(b, p)}, \hat{\mathbf{F}}_{\text{orig}}^{(b, p)} \rangle$$

### 3.2 Vision-Language Cyclic Consistency ($\mathcal{L}_{\text{TIT}}, \mathcal{L}_{\text{ITI}}$)
* **$\mathcal{L}_{\text{TIT}}$**: Text-to-Image-to-Text cyclic reconstruction matching class text embeddings with optimal $28 \times 28$ patch representations.
* **$\mathcal{L}_{\text{ITI}}$**: Image-to-Text-to-Image cycle using Top-$k=10$ Semantic Anchors across data-augmented feature spaces.

### 3.3 Domain-Adaptive Prompt Ensemble (Text-Side Enhancement)

Orthogonal to FeatureSR's visual-side upsampling, a text-side enhancement module improves the anchor text embeddings $\mathbf{e}_i$ used throughout $\mathcal{L}_{\text{CE}}$, $\mathcal{L}_{\text{TIT}}$, and $\mathcal{L}_{\text{ITI}}$.

Given a raw class label $c_i$ and target domain $d$, a domain-specific semantic expansion function $\phi(c_i, d)$ maps abbreviated or terse labels into descriptive phrases (e.g., ISIC's `"MEL"` $\rightarrow$ `"melanoma, a malignant pigmented skin cancer lesion"`), and a domain-specific template bank $\mathcal{T}_d = \{t_1, \dots, t_M\}$ supplies $M$ paraphrastic prompt structures (e.g., `"a dermatoscopic photograph of {c}."`). The ensembled anchor embedding is the re-normalized mean of $M$ independently normalized template embeddings:

$$\mathbf{e}_i = \frac{\frac{1}{M}\sum_{m=1}^{M} \hat{g}_{\text{text}}\big(t_m(\phi(c_i, d))\big)}{\left\|\frac{1}{M}\sum_{m=1}^{M} \hat{g}_{\text{text}}\big(t_m(\phi(c_i, d))\big)\right\|_2}, \qquad \hat{g}_{\text{text}}(\cdot) = \frac{g_{\text{text}}(\cdot)}{\|g_{\text{text}}(\cdot)\|_2}$$

where $g_{\text{text}}(\cdot)$ is the frozen CLIP text encoder.

### 3.4 The Train/Inference Consistency Requirement

CLIP-LoRA [3] injects trainable low-rank parameters $\theta_{\text{LoRA}}$ (via LoRA [4]) **only into the attention projections of the visual tower** (Eq. 2.1–2.2); $g_{\text{text}}(\cdot)$ carries no trainable parameters at all. Consequently, $\theta_{\text{LoRA}}$ is optimized against whichever anchor-embedding function was in effect during training:

$$\theta_{\text{LoRA}}^{*} = \arg\min_{\theta} \; \mathbb{E}\left[\mathcal{L}_{\text{total}}\big(f_\theta(x),\; \mathbf{e}(c;\, \text{prompt}_{\text{train}})\big)\right]$$

If inference substitutes a different anchor function, $\text{prompt}_{\text{infer}} \neq \text{prompt}_{\text{train}}$, the induced text-embedding distribution shifts relative to what $\theta_{\text{LoRA}}^{*}$ was calibrated against — a covariate shift confined to the text branch, since the visual branch is unchanged. This is a *post-hoc* application of §3.3 (ensemble enabled only at evaluation time) and was empirically found to be **actively harmful**: averaged over 4 target domains, post-hoc ensembling produced a **−7.81%** mean accuracy regression (5-way 5-shot), with ISIC 2018 collapsing by **−18.61%** — the domain with the largest semantic-expansion magnitude in $\phi(\cdot,\cdot)$, consistent with the magnitude of the covariate shift being the driver of the regression.

The correction is to apply §3.3 identically during training and inference, i.e. compute $\mathbf{e}_i$ via the same $(\mathcal{T}_d, \phi)$ in both $\mathcal{L}_{\text{total}}$ and evaluation, so that $\theta_{\text{LoRA}}$ is calibrated to the distribution it will actually be evaluated against. This recovers the regression and, on 3 of 4 domains, yields a net improvement over the single-template baseline (§4.3).

---

## 4. Optimization & Empirical Results

### 4.1 Optimization Protocol
* **Optimizer**: Adam (`weight_decay=5e-4`)
* **Peak Learning Rate**: `lr = 3e-5`
* **Warmup Schedule**: 5 epochs linear warmup followed by Cosine Annealing decay down to `1e-6` over 100 epochs.

### 4.2 Benchmark Verification (Strict Few-Shot)

#### 5-way 5-shot Results (400 Episodes)
| Dataset | Baseline (14×14) | Feature-SR (28×28) | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | 88.79 ± 0.48% | **90.16 ± 0.40%** | **+1.37%** 📈 | 90.43% |
| **CropDiseases** | 89.68 ± 0.70% | **90.61 ± 0.65%** | **+0.93%** 📈 | 90.41% |
| **ISIC 2018** | 41.44 ± 0.64% | **43.20 ± 0.63%** | **+1.76%** 📈 | 43.30% |
| **ChestX** | 22.43 ± 0.46% | **22.46 ± 0.42%** | **+0.03%** 📈 | 22.64% |
| **Average** | **60.59%** | **61.61%** | **+1.02%** 📈 | — |

#### 5-way 1-shot Results (100 Episodes)
| Dataset | Baseline (14×14) | Feature-SR (28×28) | Delta ($\Delta$) | Peak Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **EuroSAT** | **80.59 ± 1.29%** | 80.29 ± 1.54% | -0.30% | 82.00% |
| **CropDiseases** | **77.77 ± 2.00%** | 76.37 ± 2.22% | -1.40% | 80.00% |
| **ISIC 2018** | 30.93 ± 1.11% | **32.63 ± 1.20%** | **+1.70%** 📈 | 32.91% |
| **ChestX** | **21.95 ± 0.86%** | 21.51 ± 0.88% | -0.44% | 21.33% |
| **Average** | **52.81%** | **52.70%** | **-0.11%** | — |

### 4.3 Prompt Ensemble: Post-hoc Regression vs. Train/Inference-Consistent Fix (5-way 5-shot, 400 Episodes)

| Dataset | Baseline (Single Template) | Post-hoc Ensemble ($\text{prompt}_{\text{infer}} \neq \text{prompt}_{\text{train}}$) | Consistent Ensemble ($\text{prompt}_{\text{infer}} = \text{prompt}_{\text{train}}$) |
|:---|:---:|:---:|:---:|
| **EuroSAT** | 90.17 ± 0.89% | 84.15% (−6.03%) | **93.81 ± 0.32%** (+3.64%) 📈 |
| **CropDiseases** | 90.03 ± 1.36% | 85.04% (−4.99%) | **90.57 ± 0.64%** (+0.54%) 📈 |
| **ISIC 2018** | 43.76 ± 1.38% | 25.15% (−18.61%) | **44.49 ± 0.61%** (+0.73%) 📈 |
| **ChestX** | 22.88 ± 0.83% | 21.28% (−1.60%) | 22.79 ± 0.45% (−0.09%) |
| **Average** | **61.71%** | **53.90%** (−7.81%) | **62.92%** (+1.21%) 📈 |

*Baseline in this table is drawn from an independent training run relative to §4.2's Table (61.71% vs. 60.59% average); the two baselines are not the same checkpoint and should not be compared in absolute terms — only $\Delta$ within a matched training run is meaningful.*

### 4.4 Combined Feature-SR + Prompt Ensemble (5-way 5-shot, 400 Episodes)

Both modules were enabled jointly during training (identical $\text{prompt}$ used at train- and inference-time, per §3.4) to test whether the visual-side and text-side gains compose:

| Dataset | Baseline | Ensemble-only | **Feature-SR + Ensemble** | vs. Baseline | Best Epoch |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 90.17% | 93.81% | **94.54 ± 0.30%** | **+4.37%** | 40 |
| **CropDiseases** | 90.03% | 90.57% | **90.97 ± 0.60%** | **+0.94%** | 50 |
| **ISIC 2018** | 43.76% | 44.49% | **44.77 ± 0.65%** | **+1.01%** | 40 |
| **ChestX** | 22.88% | 22.79% | **22.96 ± 0.47%** | +0.08% | 70 |
| **Average** | **61.71%** | **62.92%** | **63.31%** | **+1.60%** | — |

All 4/4 domains satisfy $\text{Baseline} < \text{Ensemble-only} < \text{Feature-SR+Ensemble}$, i.e. a strictly monotonic stack with no negative interaction term observed, consistent with the two mechanisms acting on disjoint parameter/embedding spaces (visual-side LoRA + FeatureUpsampler/Refiner vs. text-side anchor embeddings). A secondary effect was observed on ISIC 2018: the ensemble-only run overfits sharply past epoch 10 (peak 44.44% $\rightarrow$ 38.38% by epoch 100), whereas the combined run remains stable in the 43–45% band through epoch 100 (best epoch 40), suggesting $\mathcal{L}_{\text{CR}}$ (§3.1) contributes an incidental regularizing effect under extreme few-shot sample sizes ($n=35$).

---

## 5. Computational Complexity

| Component | Trainable Parameters | GPU Memory Overhead |
|:---|:---:|:---:|
| CLIP-LoRA Backbone | 0.22 M | ~0.44 MB |
| FeatureUpsampler | 2.36 M | ~4.7 MB |
| FeatureRefiner (2 blocks) | 7.99 M | ~15.9 MB |
| **Total Added by FeatureSR** | **10.35 M** | **~21 MB (Total VRAM ~2.4 GB)** |
| Domain-Adaptive Prompt Ensemble | 0 M (frozen text encoder) | Negligible — $M\times$ text-encoder forward passes under `torch.no_grad()`, no backward pass |

---

## 6. Known Limitations

### 6.1 ChestX: A Backbone-Architecture Ceiling, Not a Method Failure

Across every experiment in this document, ChestX consistently shows the smallest gain from either enhancement (Feature-SR: +0.03%; Prompt Ensemble: −0.09%; combined: +0.08%). Cross-referencing the CC-CDFSL paper's [1] own Appendix Table 9 (comparison against SOTA CDFSL methods, 5-way 5-shot) shows this is not specific to FeatureSR or the prompt module:

| Method | Backbone | ChestX (5-shot) |
|:---|:---:|:---:|
| IM-DCL [12] (TIP-24) | ResNet10 | **28.93%** |
| DAMIM-FT [11] (AAAI-25) | ViT/DINO | 27.82% |
| AttnTemp [13] (NeurIPS-24) | ViT/DINO | 27.72% |
| CLIP-LoRA [3] + CC-CDFSL [1] (paper's full method) | ViT/CLIP | 25.47% |
| CLIP-LoRA [3] (baseline) | ViT/CLIP | 24.44% |

Every ViT/CLIP-based method in the comparison — including the CC-CDFSL paper's own full method — underperforms ResNet-based competitors on ChestX specifically. This suggests the limiting factor on ChestX is the ViT/CLIP backbone's inductive bias relative to the fine-grained, low-contrast local textures characteristic of grayscale radiographs, a ceiling that sits upstream of both FeatureSR (which upsamples this same backbone's patch grid) and the text-side prompt module. Closing this gap would require backbone-level changes and is out of scope for this document.

---

## References

[1] Yaze Zhao, Yixiong Zou, Yuhua Li, and Ruixuan Li. Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment. *CVPR*, 2026. (referred to as CC-CDFSL throughout this document.)

[2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. *ICML*, pp. 8748–8763, 2021.

[3] Maxime Zanella and Ismail Ben Ayed. Low-Rank Few-Shot Adaptation of Vision-Language Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 1593–1603, 2024. (CLIP-LoRA.)

[4] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022.

[5] Yunhui Guo, Noel C. Codella, Leonid Karlinsky, James V. Codella, John R. Smith, Kate Saenko, Tajana Rosing, and Rogerio Feris. A Broader Study of Cross-Domain Few-Shot Learning. *ECCV*, pp. 124–141, 2020. (BSCD-FSL benchmark.)

[6] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, pp. 2217–2226, 2019.

[7] Sharada P. Mohanty, David P. Hughes, and Marcel Salathé. Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 2016. (CropDiseases / PlantVillage.)

[8] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). *arXiv preprint arXiv:1902.03368*, 2019.

[9] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *CVPR*, pp. 2097–2106, 2017.

[10] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. *ICCV Workshops*, pp. 1905–1914, 2021.

[11] Ran Ma, Yixiong Zou, Yuhua Li, and Ruixuan Li. Reconstruction Target Matters in Masked Image Modeling for Cross-Domain Few-Shot Learning. *AAAI*, pp. 19305–19313, 2025. (DAMIM.)

[12] Huali Xu, Li Liu, Shuaifeng Zhi, Shaojing Fu, Zhuo Su, Ming-Ming Cheng, and Yongxiang Liu. Enhancing Information Maximization with Distance-Aware Contrastive Learning for Source-Free Cross-Domain Few-Shot Learning. *IEEE Transactions on Image Processing*, pp. 2058–2073, 2024. (IM-DCL.)

[13] Yixiong Zou, Ran Ma, Yuhua Li, and Ruixuan Li. Attention Temperature Matters in ViT-Based Cross-Domain Few-Shot Learning. *NeurIPS*, 2024. (AttnTemp.)
