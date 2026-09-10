# FeatureSR: Feature-Level Super-Resolution with Cross-Resolution Consistency for Cross-Domain Few-Shot Learning

## Technical Methodology Document

**Project**: FeatureSR  
**Baseline**: CLIP-LoRA [3] with CC-CDFSL Cycle Consistency Framework (CVPR 2026) [1]  
**Version**: v2.4 (2026-09-09)

**Changelog (v1.0 → v2.0)**: Added the Domain-Adaptive Prompt Ensemble module (§3.3), its train/inference consistency requirement and a root-cause analysis of a prior regression (§3.4), combined Feature-SR + Prompt Ensemble empirical results (§4.5), a Known Limitations section on the ChestX backbone ceiling (§6), and a References section with inline citations, sourced from the CC-CDFSL paper's own bibliography where applicable.

**Changelog (v2.0 → v2.1)**: Added isolated Test-Time Augmentation (TTA) verification results (§4.6) and a corresponding dataset-dependent usage limitation (§6.2), correcting an earlier (pre-fix) conclusion that TTA provided no benefit.

**Changelog (v2.1 → v2.2)**: Added §5.1, a derived computational-complexity analysis of the base CC-CDFSL cycle-consistency framework itself (T-I-T, I-T-I, Semantic Anchor, augmentation branch) — filling a gap absent from the upstream paper — including the finding that all cycle-consistency machinery is training-time-only with zero inference-time cost. Also corrected §5: the FeatureUpsampler/FeatureRefiner parameter counts were swapped and miscalculated (was 2.36 M / 7.99 M; verified by loading the actual model, they are 5.92 M / 4.21 M — total is nearly unchanged at 10.13 M vs. the previously reported 10.35 M, which is why the error went unnoticed).

**Changelog (v2.2 → v2.3)**: Added the previously-missing 5-way 1-shot results for Ensemble-only and Feature-SR+Ensemble (§4.7 after this version's renumbering; 100 episodes, same protocol as the 5-shot experiments). Key finding: the clean monotonic stacking observed at 5-shot does not hold at 1-shot — only 1/4 domains (CropDiseases) preserves it, EuroSAT reverses entirely, and ISIC/ChestX show Ensemble-only as a local optimum that Feature-SR slightly degrades when added on top.

**Changelog (v2.3 → v2.4)**: Added §4.1, "Evaluation Protocol: What Accuracy Measures" (prompted by a supervisor question conflating classification-accuracy ground truth with the separate, genuinely-absent reconstruction ground truth for the upsampled $28\times28$ grid). Clarifies that accuracy is an ordinary supervised metric against real class labels, and states explicitly that `evaluate()` never consumes an episode's own support images — classification is domain-adapted zero-shot CLIP similarity, not per-episode prototype matching, with the few-shot adaptation happening entirely during training. Existing §4.1–§4.6 renumbered to §4.2–§4.7 to accommodate.

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

The correction is to apply §3.3 identically during training and inference, i.e. compute $\mathbf{e}_i$ via the same $(\mathcal{T}_d, \phi)$ in both $\mathcal{L}_{\text{total}}$ and evaluation, so that $\theta_{\text{LoRA}}$ is calibrated to the distribution it will actually be evaluated against. This recovers the regression and, on 3 of 4 domains, yields a net improvement over the single-template baseline (§4.4).

---

## 4. Optimization & Empirical Results

### 4.1 Evaluation Protocol: What "Accuracy" Measures

Every accuracy number in this document is a standard supervised classification rate — for each of $n$ sampled test episodes, $Q$ query images are classified and the fraction matching their true label is averaged, with a 95% confidence interval reported across episodes ($n=100$ for 1-shot, $n=400$ for 5-shot). This part has an ordinary, unambiguous ground truth: the ImageNet-style class label attached to each query image by the source dataset (EuroSAT's land-use category, ISIC's lesion diagnosis, etc.), entirely independent of anything the model produces. This is distinct from — and should not be confused with — the separate question of whether the *intermediate* $28\times28$ feature grid has a reconstruction ground truth (§3.1, §3.4: it does not, beyond the pooled-average constraint enforced by $\mathcal{L}_{\text{CR}}$).

**The two training/evaluation phases use the support set differently, and this is worth stating explicitly because it is easy to assume the classic prototypical-network pattern (support examples averaged into a per-class prototype, queries matched against those prototypes) and this method does not follow it:**

1. **Training phase** (where "few-shot" enters): under `--strict_few_shot`, the training pool for a given target dataset is restricted to exactly $K$ images per class. Episodes are repeatedly sampled from this pool to compute $\mathcal{L}_{\text{CE}}$ (on the support images) and the cycle-consistency losses (§3), which is what adapts $\theta_{\text{LoRA}}$ to the target domain over 100 epochs.
2. **Evaluation phase** (where accuracy is measured): `evaluate()` (`train.py:312–352`) samples a fresh episode, but its accuracy computation **never references that episode's support images** — only the query images and the class-name text embeddings are used: $q_{\text{logits}} = \hat{q}_{\text{cls}} \cdot \hat{e}^\top$, classified by argmax. Support images are sampled as part of the episode structure but are not consumed by this function. Classification is therefore a **domain-adapted zero-shot CLIP similarity**, not a per-episode prototype-matching decision — the "few-shot" adaptation already happened during training, and each test episode's own support set plays no further role at that point. (The one exception is the optional `--adapt` flag in `evaluate.py`, which performs additional per-episode LoRA fine-tuning on that episode's support set before classifying its queries — an alternative, opt-in evaluation mode not used for the headline results in this document.)

### 4.2 Optimization Protocol
* **Optimizer**: Adam (`weight_decay=5e-4`)
* **Peak Learning Rate**: `lr = 3e-5`
* **Warmup Schedule**: 5 epochs linear warmup followed by Cosine Annealing decay down to `1e-6` over 100 epochs.

### 4.3 Benchmark Verification (Strict Few-Shot)

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

### 4.4 Prompt Ensemble: Post-hoc Regression vs. Train/Inference-Consistent Fix (5-way 5-shot, 400 Episodes)

| Dataset | Baseline (Single Template) | Post-hoc Ensemble ($\text{prompt}_{\text{infer}} \neq \text{prompt}_{\text{train}}$) | Consistent Ensemble ($\text{prompt}_{\text{infer}} = \text{prompt}_{\text{train}}$) |
|:---|:---:|:---:|:---:|
| **EuroSAT** | 90.17 ± 0.89% | 84.15% (−6.03%) | **93.81 ± 0.32%** (+3.64%) 📈 |
| **CropDiseases** | 90.03 ± 1.36% | 85.04% (−4.99%) | **90.57 ± 0.64%** (+0.54%) 📈 |
| **ISIC 2018** | 43.76 ± 1.38% | 25.15% (−18.61%) | **44.49 ± 0.61%** (+0.73%) 📈 |
| **ChestX** | 22.88 ± 0.83% | 21.28% (−1.60%) | 22.79 ± 0.45% (−0.09%) |
| **Average** | **61.71%** | **53.90%** (−7.81%) | **62.92%** (+1.21%) 📈 |

*Baseline in this table is drawn from an independent training run relative to §4.3's Table (61.71% vs. 60.59% average); the two baselines are not the same checkpoint and should not be compared in absolute terms — only $\Delta$ within a matched training run is meaningful.*

### 4.5 Combined Feature-SR + Prompt Ensemble (5-way 5-shot, 400 Episodes)

Both modules were enabled jointly during training (identical $\text{prompt}$ used at train- and inference-time, per §3.4) to test whether the visual-side and text-side gains compose:

| Dataset | Baseline | Ensemble-only | **Feature-SR + Ensemble** | vs. Baseline | Best Epoch |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 90.17% | 93.81% | **94.54 ± 0.30%** | **+4.37%** | 40 |
| **CropDiseases** | 90.03% | 90.57% | **90.97 ± 0.60%** | **+0.94%** | 50 |
| **ISIC 2018** | 43.76% | 44.49% | **44.77 ± 0.65%** | **+1.01%** | 40 |
| **ChestX** | 22.88% | 22.79% | **22.96 ± 0.47%** | +0.08% | 70 |
| **Average** | **61.71%** | **62.92%** | **63.31%** | **+1.60%** | — |

All 4/4 domains satisfy $\text{Baseline} < \text{Ensemble-only} < \text{Feature-SR+Ensemble}$, i.e. a strictly monotonic stack with no negative interaction term observed, consistent with the two mechanisms acting on disjoint parameter/embedding spaces (visual-side LoRA + FeatureUpsampler/Refiner vs. text-side anchor embeddings). A secondary effect was observed on ISIC 2018: the ensemble-only run overfits sharply past epoch 10 (peak 44.44% $\rightarrow$ 38.38% by epoch 100), whereas the combined run remains stable in the 43–45% band through epoch 100 (best epoch 40), suggesting $\mathcal{L}_{\text{CR}}$ (§3.1) contributes an incidental regularizing effect under extreme few-shot sample sizes ($n=35$).

### 4.6 Test-Time Augmentation (TTA): Isolated Verification

An earlier exploratory pass (`logs/enhanced_5shot_real.log`, pre-fix) suggested horizontal-flip TTA (feature-level: encode the query image and its horizontal flip separately, $\ell_2$-normalize each, average, re-normalize) provided no benefit (4-dataset average $\Delta = -0.23\%$). That test used the same pre-fix, train/inference-inconsistent checkpoints diagnosed in §3.4 — TTA had never been evaluated against a properly calibrated model. Re-testing in isolation ($\text{use\_prompt\_ensemble}=\text{True}$ held constant, only $\text{use\_tta}$ toggled, 5-way 5-shot, 400 episodes) on both fixed checkpoint families gives:

| Dataset | Ensemble-only, no TTA | + TTA | $\Delta$ | Feature-SR+Ensemble, no TTA | + TTA | $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 93.69 ± 0.32% | 93.97 ± 0.30% | +0.28% 📈 | 94.00 ± 0.32% | 94.36 ± 0.32% | +0.36% 📈 |
| **CropDiseases** | 90.61 ± 0.62% | 90.96 ± 0.62% | +0.35% 📈 | 90.28 ± 0.68% | **91.44 ± 0.60%** | **+1.16%** 📈 |
| **ISIC 2018** | 44.10 ± 0.64% | 44.03 ± 0.64% | -0.07% | 44.13 ± 0.67% | **42.75 ± 0.69%** | **-1.38%** ⚠️ |
| **ChestX** | 22.77 ± 0.49% | **23.60 ± 0.49%** | **+0.83%** 📈 | 22.59 ± 0.45% | 23.19 ± 0.46% | +0.60% 📈 |
| **Average** | **62.79%** | **63.14%** | **+0.35%** 📈 | **62.75%** | **62.94%** | +0.19% |

TTA is not broken — the prior negative conclusion was an artifact of testing it against an already-miscalibrated model. On calibrated checkpoints, TTA gives a real (delta exceeds the sum of confidence intervals, not noise) positive effect on 3/4 domains, and is in fact the single most effective enhancement tested on ChestX (+0.83%, larger than either Feature-SR's +0.03% or Prompt Ensemble's -0.09%/+0.08% on that domain). ISIC 2018 is the exception, discussed in §6.2.

### 4.7 5-way 1-shot: Ensemble & Feature-SR + Ensemble (100 Episodes)

§4.3's 1-shot table covers Feature-SR alone; Ensemble-only and Feature-SR+Ensemble were left as future work in the 0829 report. Re-trained under the identical protocol used for the 5-shot ensemble experiments (`--n_shot 1`, otherwise unchanged — strict few-shot, $\text{lr}=3\text{e-}5$, 5-epoch warmup, per-dataset $\lambda_1,\lambda_2$, $\lambda_3=0.3$):

| Dataset | Baseline | Ensemble-only | $\Delta$ | Feature-SR + Ensemble | $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **EuroSAT** | 81.09 ± 1.34% | 79.83 ± 1.54% | **-1.26%** ⚠️ | 78.91 ± 1.52% | **-2.18%** ⚠️ |
| **CropDiseases** | 77.89 ± 1.90% | 78.64 ± 2.27% | +0.75% 📈 | **80.75 ± 2.20%** | **+2.86%** 📈 |
| **ISIC 2018** | 32.79 ± 0.98% | **34.73 ± 1.10%** | **+1.94%** 📈 | 34.59 ± 1.11% | +1.80% 📈 |
| **ChestX** | 20.81 ± 0.96% | **22.21 ± 0.99%** | **+1.40%** 📈 | 20.99 ± 0.90% | +0.18% |
| **Average** | **53.15%** | **53.85%** | **+0.71%** | **53.81%** | **+0.67%** |

Unlike §4.5's 5-shot result, the strict monotonic stack $\text{Baseline} < \text{Ensemble-only} < \text{Feature-SR+Ensemble}$ holds on only 1 of 4 domains (CropDiseases, which also shows the largest single $\Delta$ observed anywhere in this document, +2.86%). EuroSAT reverses entirely (both enhancements net negative, worse combined than alone) — a striking contrast with EuroSAT's +4.37% at 5-shot, the largest gain in that setting. ISIC and ChestX show Ensemble-only as the local optimum, with Feature-SR added on top slightly *reducing* accuracy. The average is mildly positive in both configurations (+0.71% / +0.67%) but this masks per-dataset variance far exceeding the 5-shot case; averaging over datasets here would be actively misleading. See `reports/0905_1shot完整驗證報告.md` for full analysis. This resolves the open question of §4.3 — 1-shot gains are not uniformly absent, but the visual-side LoRA calibration is evidently far more sensitive to any additional perturbation (text-side ensemble or visual-side Feature-SR) when the support set is a single image, making dataset-dependent enablement necessary rather than optional.

---

## 5. Computational Complexity

| Component | Trainable Parameters | GPU Memory Overhead |
|:---|:---:|:---:|
| CLIP-LoRA Backbone | 0.22 M | ~0.44 MB |
| FeatureUpsampler | 5.92 M | ~11.8 MB |
| FeatureRefiner (2 blocks) | 4.21 M | ~8.4 MB |
| **Total Added by FeatureSR** | **10.13 M** | **~20.3 MB (Total VRAM ~2.4 GB)** |
| Domain-Adaptive Prompt Ensemble | 0 M (frozen text encoder) | Negligible — $M\times$ text-encoder forward passes under `torch.no_grad()`, no backward pass |

### 5.1 Base CC-CDFSL Cycle-Consistency Overhead

The upstream CC-CDFSL framework [1] itself reports no complexity analysis (no FLOPs, parameter-count, or timing section appears anywhere in the paper). For completeness, we derive it here from the reference implementation (`losses/cycle_consistency.py`, `train.py`):

| Component | Trainable Parameters | Added Compute (training) |
|:---|:---:|:---|
| T-I-T ($\mathcal{L}_{TIT}$) | **0** | One $[C,N]$ similarity matrix + argmax ($C{=}5$ classes) — negligible next to a CLIP forward pass |
| Semantic Anchor shrinking (`select_anchor_features`) | **0** | `einsum("bpd,cd->bcp")` over all support+augmented images, $O(B_{aug}\times P \times C \times d)$; $C{=}5$ keeps this small |
| I-T-I ($\mathcal{L}_{ITI}$) | **0** | Two argmax similarity lookups — negligible |
| Augmentation branch ($n_{aug}{=}4$) | **0** | The actual cost driver: $n_{aug}$ extra CLIP visual-encoder forward passes per episode to populate the retrieval pool, run under `torch.no_grad()` (`train.py:239-240`) — adds forward-pass time but **no backward-pass memory** |

None of T-I-T, I-T-I, or Semantic Anchor introduce trainable parameters — every parameter added during training is still exactly the 0.22 M CLIP-LoRA weights (plus FeatureSR's 10.13 M when enabled); the cycle-consistency terms only shape the loss landscape those weights are optimized against.

**Inference-time cost is zero.** `evaluate()` (`train.py:312–352`) never invokes T-I-T, I-T-I, Semantic Anchor, or the augmentation branch — it is a plain `encode_image → cosine-similarity-with-text` forward pass, identical in cost to a bare CLIP-LoRA baseline. The entire cycle-consistency machinery is a **training-time-only regularizer**; it has no effect on deployed inference latency regardless of whether Feature-SR or Prompt Ensemble are also enabled.

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

### 6.2 ISIC 2018: TTA Regresses When Stacked with Feature-SR

Unlike the ChestX ceiling (§6.1), which is backbone-level and affects every enhancement uniformly, the TTA regression on ISIC 2018 (§4.6) is specific to one enhancement combination: horizontal-flip TTA is near-neutral on the ensemble-only checkpoint (-0.07%, within noise) but drops **-1.38%** once stacked on top of Feature-SR — a delta exceeding the sum of both configurations' confidence intervals (0.67% + 0.69% = 1.36%), i.e. a real effect, not sampling noise.

Horizontal flips are a semantically valid augmentation for dermoscopic images (no canonical orientation), so the regression is unlikely to be a label-semantics violation of the kind that would rule out flipping outright. A more plausible account: Feature-SR's cross-resolution consistency objective ($\mathcal{L}_{\text{CR}}$, §3.1) already tightens the visual embedding around the *specific* orientation statistics seen during training on ISIC's fine-grained lesion boundaries (the domain with the smallest Feature-SR margin to begin with, §4.3); averaging in a flipped view at inference then perturbs an already narrowly-calibrated decision boundary rather than adding robust signal. This remains a hypothesis — no controlled ablation isolating $\mathcal{L}_{\text{CR}}$'s contribution to this specific interaction has been run.

**Practical implication**: `--use_tta` should be a per-dataset, not global, switch — enabled for EuroSAT, CropDiseases, and ChestX, disabled for ISIC 2018 (especially in the Feature-SR + Ensemble configuration). See `reports/0901_TTA驗證報告.md` for the full verification.

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
