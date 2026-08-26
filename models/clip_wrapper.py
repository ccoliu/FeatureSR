"""
models/clip_wrapper.py
CLIP 模型包裝器，提供：
  - 全域 CLS 特徵提取
  - 局部 patch 特徵提取（ViT 中間層輸出）
  - 文字特徵提取
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPWrapper(nn.Module):
    """
    CLIP 模型包裝器。

    核心功能：
    1. encode_image_with_patches()：
       同時返回 CLS token 和所有 patch tokens（局部特徵）
    2. encode_text()：
       提取文字特徵
    """

    def __init__(self, backbone: str = "ViT-B/16", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(backbone, device=device)
        self.model = self.model.float()  # 轉為 float32 方便訓練

        # 取得模型尺寸
        if hasattr(self.model.visual, "transformer"):
            # ViT architecture
            self.embed_dim = self.model.visual.transformer.width
            # CLIP ViT 用 conv1 做 patch embedding，kernel_size = patch_size
            patch_size = self.model.visual.conv1.kernel_size[0]
            self.n_patches = (self.model.visual.input_resolution // patch_size) ** 2
        else:
            # ResNet architecture：使用 attnpool 前的 global pool 特徵
            self.embed_dim = self.model.visual.attnpool.c_proj.out_features
            self.n_patches = None

        self.output_dim = self.model.visual.output_dim

        # 凍結所有參數（將由 LoRA 或其他 PEFT 層解凍）
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def encode_image_with_patches(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取圖像的 CLS 特徵和 patch 局部特徵。

        Args:
            images: [B, 3, H, W]

        Returns:
            cls_feat: [B, d]      L2 normalized CLS 特徵
            patch_feat: [B, P, d] L2 normalized patch 特徵（P = 196 for ViT-B/16）
        """
        visual = self.model.visual
        x = images.to(self.device)

        # ViT forward（截取中間輸出）
        x = visual.conv1(x)                          # [B, width, H/patch, W/patch]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, width, P]
        x = x.permute(0, 2, 1)                       # [B, P, width]

        # 加入 class token 和 positional embedding
        cls_tokens = visual.class_embedding.unsqueeze(0).unsqueeze(0)
        cls_tokens = cls_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)        # [B, P+1, width]
        x = x + visual.positional_embedding

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)                       # [P+1, B, width]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)                       # [B, P+1, width]

        # 分離 CLS token 和 patch tokens
        cls_token = x[:, 0, :]                       # [B, width]
        patch_tokens = x[:, 1:, :]                   # [B, P, width]

        # Layer norm + projection
        cls_token = visual.ln_post(cls_token)
        if visual.proj is not None:
            cls_feat = cls_token @ visual.proj        # [B, d]
        else:
            cls_feat = cls_token

        # 對 patch tokens 應用 ln_post + proj
        patch_tokens = visual.ln_post(patch_tokens)  # [B, P, width]
        if visual.proj is not None:
            patch_feat = patch_tokens @ visual.proj  # [B, P, d]
        else:
            patch_feat = patch_tokens

        # L2 normalize
        cls_feat = F.normalize(cls_feat, dim=-1)
        patch_feat = F.normalize(patch_feat, dim=-1)

        return cls_feat, patch_feat

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """
        用類別名稱生成文字特徵（使用 "a photo of a {class_name}" 模板）。

        Args:
            class_names: 類別名稱列表，長度 C

        Returns:
            text_feat: [C, d] L2 normalized 文字特徵
        """
        # 使用論文中提及的通用 prompt template
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)   # [C, d]

        text_feat = F.normalize(text_feat.float(), dim=-1)
        return text_feat

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """僅提取 CLS 特徵（用於推論）"""
        cls_feat, _ = self.encode_image_with_patches(images)
        return cls_feat

    def logit_scale(self) -> float:
        return self.model.logit_scale.exp().item()
