"""
models/feature_sr.py
Feature-level Super-Resolution Module for CC-CDFSL

核心想法：
  不在 pixel 空間做 SR（已證明會引入 domain-specific artifacts），
  而是在 CLIP 的 patch feature 空間做「超解析度」，
  將 14×14 = 196 個 patches 上採樣到 28×28 = 784 個 patches，
  提供更精細的局部特徵給 Cycle Consistency matching。

架構：
  FeatureUpsampler:    bilinear interpolation + learnable residual
  FeatureRefiner:      輕量 Transformer blocks 精煉 upsampled features
  FeatureSRModule:     整合以上兩者的包裝器

VRAM 估算（ViT-B/16, batch_size=25, d=512）：
  原始 patches:        25 × 196 × 512 × 4B ≈ 10 MB
  上採樣到 28×28:      25 × 784 × 512 × 4B ≈ 40 MB
  Refiner (2 layers):  ~2M params × 4B ≈ 8 MB
  總增量:              ~60 MB（對 3060 的 12GB 完全沒壓力）
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureUpsampler(nn.Module):
    """
    Feature-level 上採樣模組。

    將 ViT patch features 從 (H, W) 上採樣到 (scale*H, scale*W)。
    使用 bilinear interpolation 作為基礎 + learnable residual 精煉。

    方法：
      1. 將 [B, P, d] reshape 為 [B, H, W, d]（H=W=14 for ViT-B/16）
      2. Bilinear interpolate 到 [B, sH, sW, d]
      3. 加上 learnable residual correction
      4. 展平回 [B, sH*sW, d]

    Args:
        feat_dim:    特徵維度 (d=512 for ViT-B/16)
        input_size:  輸入 patch grid 大小 (14 for ViT-B/16)
        scale:       上採樣倍率 (預設 2，即 14→28)
        use_residual: 是否使用 learnable residual（建議 True）
    """

    def __init__(
        self,
        feat_dim: int = 512,
        input_size: int = 14,
        scale: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_size = input_size
        self.scale = scale
        self.output_size = input_size * scale
        self.use_residual = use_residual

        if use_residual:
            # Learnable residual: 1×1 conv → GELU → 1×1 conv
            # 在 feature map 上做局部精煉
            self.residual_net = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([feat_dim, self.output_size, self.output_size]),
                nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            )
            # 初始化為接近零，讓初期行為接近純 bilinear
            self._init_residual_near_zero()

        # Learnable positional embedding for upsampled positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.output_size * self.output_size, feat_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_residual_near_zero(self):
        """初始化 residual network 的權重接近零，確保訓練初期不破壞原始特徵"""
        for m in self.residual_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_feat: torch.Tensor) -> torch.Tensor:
        """
        上採樣 patch features。

        Args:
            patch_feat: [B, P, d] 其中 P = input_size^2 (196 for 14×14)

        Returns:
            upsampled: [B, P_up, d] 其中 P_up = output_size^2 (784 for 28×28)
        """
        B, P, d = patch_feat.shape
        H = W = self.input_size

        assert P == H * W, f"Expected P={H*W}, got P={P}"
        assert d == self.feat_dim, f"Expected d={self.feat_dim}, got d={d}"

        # Step 1: Reshape 為 2D feature map
        feat_2d = patch_feat.view(B, H, W, d).permute(0, 3, 1, 2)  # [B, d, H, W]

        # Step 2: Bilinear interpolation
        feat_up = F.interpolate(
            feat_2d,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False,
        )  # [B, d, sH, sW]

        # Step 3: Learnable residual correction
        if self.use_residual:
            residual = self.residual_net(feat_up)  # [B, d, sH, sW]
            feat_up = feat_up + residual

        # Step 4: Reshape back to sequence
        feat_up = feat_up.permute(0, 2, 3, 1)  # [B, sH, sW, d]
        feat_up = feat_up.reshape(B, -1, d)     # [B, sH*sW, d]

        # Step 5: Add positional embedding
        feat_up = feat_up + self.pos_embed

        return feat_up


class FeatureRefinerBlock(nn.Module):
    """
    輕量 Transformer block，用於精煉 upsampled features。

    使用 pre-norm 架構，與 CLIP ViT 的風格一致。

    Args:
        feat_dim:  特徵維度
        n_heads:   attention heads 數量
        mlp_ratio: MLP 隱藏層倍率
        dropout:   dropout rate
    """

    def __init__(
        self,
        feat_dim: int = 512,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.ln1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        mlp_hidden = int(feat_dim * mlp_ratio)
        self.ln2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feat_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d] patch features

        Returns:
            x: [B, N, d] refined features
        """
        # Pre-norm self-attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.ln2(x))

        return x


class FeatureRefiner(nn.Module):
    """
    由多層 FeatureRefinerBlock 組成的精煉網路。

    注意：為了控制計算量（特別是 28×28=784 tokens 的 self-attention），
    使用較少的層數和較小的 MLP ratio。

    VRAM 估算（2 layers, d=512, 784 tokens, B=25）：
      Self-attention: 25 × 784 × 784 × 4B ≈ 61 MB（可接受）
      Parameters:     ~2M × 4B ≈ 8 MB
      總增量:          ~70 MB

    Args:
        feat_dim:   特徵維度
        n_layers:   Transformer block 數量
        n_heads:    attention heads 數量
        mlp_ratio:  MLP 隱藏層倍率
        dropout:    dropout rate
    """

    def __init__(
        self,
        feat_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            FeatureRefinerBlock(
                feat_dim=feat_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # 最終 layer norm
        self.ln_post = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d] upsampled patch features

        Returns:
            x: [B, N, d] refined features
        """
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class FeatureSRModule(nn.Module):
    """
    完整的 Feature-level SR 模組，整合 Upsampler + Refiner。

    Pipeline:
      CLIP patches [B, 196, 512]
        → FeatureUpsampler → [B, 784, 512]
        → FeatureRefiner   → [B, 784, 512]
        → L2 normalize     → [B, 784, 512]

    同時保留原始 14×14 patches 用於 Cross-Resolution Consistency Loss。

    Args:
        feat_dim:       CLIP feature 維度 (512 for ViT-B/16)
        input_size:     原始 patch grid 大小 (14 for ViT-B/16)
        scale:          上採樣倍率 (2 → 28×28, 4 → 56×56)
        refiner_layers: Refiner Transformer block 數量
        refiner_heads:  Refiner attention heads
        refiner_mlp_ratio: Refiner MLP 隱藏層倍率
        dropout:        Dropout rate
        use_residual:   Upsampler 是否使用 learnable residual
    """

    def __init__(
        self,
        feat_dim: int = 512,
        input_size: int = 14,
        scale: int = 2,
        refiner_layers: int = 2,
        refiner_heads: int = 8,
        refiner_mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.input_size = input_size
        self.scale = scale
        self.output_size = input_size * scale

        self.upsampler = FeatureUpsampler(
            feat_dim=feat_dim,
            input_size=input_size,
            scale=scale,
            use_residual=use_residual,
        )

        self.refiner = FeatureRefiner(
            feat_dim=feat_dim,
            n_layers=refiner_layers,
            n_heads=refiner_heads,
            mlp_ratio=refiner_mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        patch_feat: torch.Tensor,
        return_both: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Feature-level SR。

        Args:
            patch_feat:  [B, P, d] 原始 CLIP patch features (P=196, d=512)
            return_both: 若 True，同時返回原始和上採樣特徵（用於 Cross-Resolution Loss）

        Returns:
            sr_feat:     [B, P_up, d] L2-normalized 上採樣 patch features
            orig_feat:   [B, P, d] 原始 patch features（僅當 return_both=True）
        """
        # 上採樣 + 精煉
        upsampled = self.upsampler(patch_feat)        # [B, P_up, d]
        refined = self.refiner(upsampled)             # [B, P_up, d]

        # L2 normalize（與 CLIP 特徵空間一致）
        sr_feat = F.normalize(refined, dim=-1)        # [B, P_up, d]

        if return_both:
            return sr_feat, patch_feat
        return sr_feat, None

    def get_num_output_patches(self) -> int:
        """返回上採樣後的 patch 數量"""
        return self.output_size ** 2

    def count_parameters(self) -> dict:
        """統計參數量"""
        upsampler_params = sum(p.numel() for p in self.upsampler.parameters())
        refiner_params = sum(p.numel() for p in self.refiner.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "upsampler": upsampler_params,
            "refiner": refiner_params,
            "trainable": trainable,
            "total": total,
        }


class CrossResolutionConsistencyLoss(nn.Module):
    """
    Cross-Resolution Consistency Loss (L_feat_sr)。

    確保上採樣後的 features（pooled back 到原始解析度）
    與原始 features 之間保持語義一致性。

    這是一個自監督正則化項：
      sr_feat [B, P_up, d] → adaptive_avg_pool → [B, P, d]
      L_feat_sr = 1 - mean(cosine_sim(pooled_sr, orig))

    防止 Feature SR Module 在上採樣過程中偏離 CLIP 的語義空間。

    Args:
        input_size:  原始 patch grid 大小 (14)
        scale:       上採樣倍率 (2)
    """

    def __init__(self, input_size: int = 14, scale: int = 2):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size * scale

    def forward(
        self,
        sr_feat: torch.Tensor,    # [B, P_up, d] 上採樣特徵
        orig_feat: torch.Tensor,  # [B, P, d] 原始特徵
    ) -> torch.Tensor:
        """
        計算 cross-resolution consistency loss。

        Args:
            sr_feat:   [B, P_up, d] L2-normalized 上採樣 patch features
            orig_feat: [B, P, d] L2-normalized 原始 patch features

        Returns:
            loss: scalar tensor
        """
        B, P_up, d = sr_feat.shape
        _, P, _ = orig_feat.shape
        H_up = W_up = self.output_size
        H = W = self.input_size

        # Step 1: 將 sr_feat reshape 為 2D
        sr_2d = sr_feat.view(B, H_up, W_up, d).permute(0, 3, 1, 2)  # [B, d, H_up, W_up]

        # Step 2: Adaptive average pool 回原始解析度
        pooled = F.adaptive_avg_pool2d(sr_2d, (H, W))  # [B, d, H, W]
        pooled = pooled.permute(0, 2, 3, 1).reshape(B, P, d)  # [B, P, d]

        # Step 3: L2 normalize pooled features
        pooled = F.normalize(pooled, dim=-1)

        # Step 4: Cosine similarity（因為都已 L2 normalized，dot product = cosine sim）
        # 確保 orig_feat 也是 L2 normalized
        orig_norm = F.normalize(orig_feat, dim=-1)

        # 逐 patch 計算 cosine similarity
        sim = (pooled * orig_norm).sum(dim=-1)  # [B, P]
        loss = 1.0 - sim.mean()

        return loss
