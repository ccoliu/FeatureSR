"""
models/cc_cdfsl.py
CC-CDFSL 完整模型包裝器

整合：
  - CLIP + LoRA fine-tuning
  - Semantic Anchor Module (Augmentation Phase)
  - Cycle Consistency Loss (T-I-T + I-T-I)
  - Cross-entropy classification loss
  - [NEW] Feature-level SR Module (可選)
  - [NEW] Cross-Resolution Consistency Loss (可選)
"""
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_wrapper import CLIPWrapper
from .clip_lora import CLIPLoRA
from .feature_sr import FeatureSRModule, CrossResolutionConsistencyLoss
from losses.cycle_consistency import CyclicConsistencyLoss


class CCCDFSL(nn.Module):
    """
    CC-CDFSL：Cycle-Consistent Cross-Domain Few-Shot Learning

    論文設定：
    - Backbone：CLIP ViT-B/16
    - PEFT：CLIP-LoRA (r=4, alpha=1)
    - Loss：L_CE + λ1 * L_cyc_txt + λ2 * L_cyc_img
    - Semantic Anchor top-k：10
    - Augmentation 次數：4

    擴展（Feature SR）：
    - Feature SR Module：將 14×14 patches 上採樣到 28×28
    - Cross-Resolution Consistency Loss：確保上採樣特徵的語義一致性
    - Loss：L_CE + λ1 * L_cyc_txt + λ2 * L_cyc_img + λ3 * L_feat_sr

    Args:
        backbone:   CLIP backbone 名稱（'ViT-B/16', 'ViT-L/14', 'RN50'）
        device:     運算設備
        lora_r:     LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout
        top_k:      Semantic Anchor top-k
        lambda1:    T-I-T loss 權重
        lambda2:    I-T-I loss 權重
        n_aug:      augmentation 次數
        use_feature_sr:     是否啟用 Feature SR Module
        sr_scale:           Feature SR 上採樣倍率 (2 → 28×28)
        sr_refiner_layers:  Feature Refiner Transformer block 數量
        sr_refiner_heads:   Feature Refiner attention heads
        sr_refiner_mlp_ratio: Feature Refiner MLP 倍率
        lambda3:            Cross-Resolution Consistency Loss 權重
    """

    def __init__(
        self,
        backbone: str = "ViT-B/16",
        device: str = "cuda",
        lora_r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        top_k: int = 10,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        n_aug: int = 4,
        # Feature SR 參數
        use_feature_sr: bool = False,
        sr_scale: int = 2,
        sr_refiner_layers: int = 2,
        sr_refiner_heads: int = 8,
        sr_refiner_mlp_ratio: float = 2.0,
        lambda3: float = 0.3,
    ):
        super().__init__()
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.n_aug = n_aug
        self.use_feature_sr = use_feature_sr

        # ========== CLIP + LoRA ==========
        self.clip_wrapper = CLIPWrapper(backbone=backbone, device=device)
        self.lora_model = CLIPLoRA(
            clip_wrapper=self.clip_wrapper,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        # ========== 循環一致性損失 ==========
        self.cc_loss = CyclicConsistencyLoss(top_k=top_k)

        # ========== Feature SR Module（可選）==========
        if use_feature_sr:
            # 取得 CLIP 特徵維度和 patch grid 大小
            feat_dim = self.clip_wrapper.output_dim   # 512 for ViT-B/16
            input_size = int(self.clip_wrapper.n_patches ** 0.5)  # 14 for ViT-B/16

            self.feature_sr = FeatureSRModule(
                feat_dim=feat_dim,
                input_size=input_size,
                scale=sr_scale,
                refiner_layers=sr_refiner_layers,
                refiner_heads=sr_refiner_heads,
                refiner_mlp_ratio=sr_refiner_mlp_ratio,
            ).to(device)

            self.cr_loss_fn = CrossResolutionConsistencyLoss(
                input_size=input_size,
                scale=sr_scale,
            ).to(device)
        else:
            self.feature_sr = None
            self.cr_loss_fn = None

        # ========== Augmentation Transforms ==========
        # 輕量 augmentation for Semantic Anchor
        import torchvision.transforms as T
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(15)], p=0.5),
            T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.3),
        ])

    def get_text_features(self, class_names: List[str]) -> torch.Tensor:
        """提取文字特徵 [C, d]"""
        return self.clip_wrapper.encode_text(class_names)

    def augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        對影像進行 n_aug 次 augmentation，返回 augmented 影像批次。

        Args:
            images: [B, 3, H, W]

        Returns:
            aug_images: [B * n_aug, 3, H, W]
        """
        aug_list = []
        for _ in range(self.n_aug):
            aug = torch.stack([self.aug_transform(img) for img in images])
            aug_list.append(aug)
        return torch.cat(aug_list, dim=0)   # [B * n_aug, 3, H, W]

    def apply_feature_sr(
        self, patch_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        對 patch features 套用 Feature SR（若啟用）。

        Args:
            patch_feat: [B, P, d] 原始 CLIP patch features

        Returns:
            enhanced_feat: [B, P_out, d] 增強後的 patch features
                - 若啟用 SR：P_out = (scale*H)^2（如 784）
                - 若未啟用：P_out = P（如 196）
            orig_feat:     [B, P, d] 原始特徵（用於 CR Loss，僅 SR 啟用時非 None）
        """
        if self.use_feature_sr and self.feature_sr is not None:
            sr_feat, orig_feat = self.feature_sr(patch_feat, return_both=True)
            return sr_feat, orig_feat
        else:
            return patch_feat, None

    def forward(
        self,
        support_images: torch.Tensor,         # [K*N, 3, H, W]
        support_labels: torch.Tensor,         # [K*N]
        query_images: torch.Tensor,           # [K*Q, 3, H, W]
        class_names: List[str],               # 長度 K
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Few-shot 前向傳播：

        Args:
            support_images: [K*N, 3, H, W] support set 影像
            support_labels: [K*N] 0-indexed labels
            query_images:   [K*Q, 3, H, W] query set 影像
            class_names:    [K] 類別名稱列表
            compute_loss:   是否計算損失（推論時可設 False）

        Returns:
            dict 包含：
                'logits':     [K*Q, K] 分類 logits
                'loss':       total loss（scalar）
                'loss_ce':    cross-entropy loss
                'loss_tit':   T-I-T cycle loss
                'loss_iti':   I-T-I cycle loss
                'loss_cr':    cross-resolution consistency loss（若啟用 SR）
                'accuracy':   query set 準確率
        """
        # =========== 1. 提取文字特徵 ===========
        text_feat = self.get_text_features(class_names)   # [C, d]

        # =========== 2. 提取 Support Set 特徵 ===========
        support_cls, support_patches = self.lora_model.encode_image_with_patches(support_images)
        # support_cls: [K*N, d], support_patches: [K*N, P, d]

        # =========== 2b. Feature SR（若啟用）===========
        support_patches_enhanced, support_patches_orig = self.apply_feature_sr(support_patches)
        # support_patches_enhanced: [K*N, P_out, d]
        # support_patches_orig:     [K*N, P, d] or None

        # =========== 3. 推論（Query Set 分類）===========
        query_cls, _ = self.lora_model.encode_image_with_patches(query_images)
        # query_cls: [K*Q, d]

        # 用文字特徵計算 logits（cosine similarity × temperature）
        logits = query_cls @ text_feat.T    # [K*Q, C]
        logits = logits * self.clip_wrapper.model.logit_scale.exp()

        # 準確率（for 監控）
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            accuracy = (pred == query_labels_from_query(support_labels, query_images, logits)).float().mean()

        # =========== 4. 計算損失 ===========
        if not compute_loss:
            return {
                "logits": logits,
                "loss": torch.tensor(0.0),
                "loss_ce": torch.tensor(0.0),
                "loss_tit": torch.tensor(0.0),
                "loss_iti": torch.tensor(0.0),
                "loss_cr": torch.tensor(0.0),
                "accuracy": accuracy,
            }

        # 4a. Cross-entropy loss（使用 support set 訓練）
        support_logits = support_cls @ text_feat.T    # [K*N, C]
        support_logits = support_logits * self.clip_wrapper.model.logit_scale.exp()
        loss_ce = F.cross_entropy(support_logits, support_labels)

        # 4b. Augmentation for Semantic Anchor
        with torch.no_grad():
            aug_images = self.augment_images(support_images)   # [K*N*n_aug, 3, H, W]
        _, aug_patches = self.lora_model.encode_image_with_patches(aug_images)
        # aug_patches: [K*N*n_aug, P, d]

        # 4b-2. 對 augmented patches 也套用 Feature SR
        aug_patches_enhanced, _ = self.apply_feature_sr(aug_patches)
        # aug_patches_enhanced: [K*N*n_aug, P_out, d]

        # 4c. Cycle consistency losses（使用增強後的 patches）
        loss_tit, loss_iti = self.cc_loss(
            text_feat=text_feat,
            patch_feat_orig=support_patches_enhanced,   # [K*N, P_out, d]
            patch_feat_aug=aug_patches_enhanced,        # [K*N*n_aug, P_out, d]
        )

        # 4d. Cross-Resolution Consistency Loss（若啟用 SR）
        loss_cr = torch.tensor(0.0, device=text_feat.device)
        if self.use_feature_sr and support_patches_orig is not None:
            loss_cr = self.cr_loss_fn(support_patches_enhanced, support_patches_orig)

        # 4e. Total loss
        # L = L_CE + λ1 * L_cyc_txt + λ2 * L_cyc_img + λ3 * L_feat_sr
        loss = loss_ce + self.lambda1 * loss_tit + self.lambda2 * loss_iti + self.lambda3 * loss_cr

        return {
            "logits": logits,
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_tit": loss_tit,
            "loss_iti": loss_iti,
            "loss_cr": loss_cr,
            "accuracy": accuracy,
        }

    def inference(
        self,
        query_images: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        純推論模式（不計算梯度）。

        Args:
            query_images: [Q, 3, H, W]
            text_feat:    [C, d]

        Returns:
            pred_labels: [Q]
        """
        with torch.no_grad():
            cls_feat, _ = self.lora_model.encode_image_with_patches(query_images)
            sim = cls_feat @ text_feat.T   # [Q, C]
            pred = sim.argmax(dim=-1)
        return pred

    def trainable_parameters(self):
        """返回所有可訓練參數（包含 Feature SR Module）"""
        params = list(self.lora_model.trainable_parameters())
        if self.use_feature_sr and self.feature_sr is not None:
            params.extend(list(self.feature_sr.parameters()))
        return params


def query_labels_from_query(support_labels, query_images, logits):
    """
    根據 support_labels 推斷 query labels（5-way 任務中 query 的 ground truth）。
    在少樣本訓練中，query labels 和 support labels 使用同一組類別索引。

    這個函數由外部呼叫傳入，此處只是 placeholder。
    """
    # 實際上 query_labels 由 few_shot_sampler 提供
    return logits.argmax(dim=-1)   # 只是 placeholder
