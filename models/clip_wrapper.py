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


# 資料夾類別名稱 → prompt 用的完整名稱，沿用 ATHA coop_lora_trainer.py 的 label_names。
# ISIC 資料夾是縮寫、EuroSAT 是黏在一起的寫法；ChestX / CropDiseases 原本就是完整名稱，不需對照。
DISPLAY_NAMES = {
    "isic": {
        "MEL": "Melanoma",
        "NV": "Melanocytic Nevus",
        "BCC": "Basal Cell Carcinoma",
        "AKIEC": "Actinic Keratosis",
        "BKL": "Benign Keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular Lesion",
    },
    "eurosat": {
        "AnnualCrop": "Annual Crop Land",
        "Forest": "Forest",
        "HerbaceousVegetation": "Herbaceous Vegetation Land",
        "Highway": "Highway or Road",
        "Industrial": "Industrial Buildings",
        "Pasture": "Pasture Land",
        "PermanentCrop": "Permanent Crop Land",
        "Residential": "Residential Buildings",
        "River": "River",
        "SeaLake": "Sea or Lake",
    },
}


def display_name(name: str, dataset_name: Optional[str]) -> str:
    return DISPLAY_NAMES.get((dataset_name or "").lower(), {}).get(name, name)


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
        self, images: torch.Tensor, patch_mode: str = "plain"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取圖像的 CLS 特徵和 patch 局部特徵。

        Args:
            images: [B, 3, H, W]
            patch_mode: "plain"：patch 取最後一層的完整輸出（舊行為）
                        "value"：MaskCLIP 式，最後一層的 patch 只取 value → out_proj，
                                 不經 attention 混合、殘差與 MLP；最後一層原始 patch 與文字的局部對齊很差。
                                 CLS 不受影響。LoRA 已注入時 value 也包含 LoRA 增量。

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
        if patch_mode == "plain":
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                   # [B, P+1, width]
            cls_token = x[:, 0, :]                   # [B, width]
            patch_tokens = x[:, 1:, :]               # [B, P, width]
        elif patch_mode == "value":
            blocks = visual.transformer.resblocks
            for blk in blocks[:-1]:
                x = blk(x)
            last = blocks[-1]
            cls_token = last(x)[0]                   # [B, width]，CLS 照常走完整的最後一層
            patch_tokens = self._value_path(last, x[1:]).permute(1, 0, 2)   # [B, P, width]
        else:
            raise ValueError(f"未知的 patch_mode: {patch_mode}")

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

    @staticmethod
    def _value_path(block, x: torch.Tensor) -> torch.Tensor:
        """block 的 value → out_proj（x: [P, B, width]），有 LoRA 時用 LoRA 版本的投影"""
        attn = block.attn
        width = x.shape[-1]
        h = block.ln_1(x).reshape(-1, width)
        in_proj = getattr(attn, "in_proj_lora", None)
        if in_proj is not None:
            v = in_proj(h)[:, 2 * width:]
        else:
            v = F.linear(h, attn.in_proj_weight[2 * width:], attn.in_proj_bias[2 * width:])
        out_proj = getattr(attn, "out_proj_lora", None)
        if out_proj is not None:
            v = out_proj(v)
        else:
            v = F.linear(v, attn.out_proj.weight, attn.out_proj.bias)
        return v.reshape(x.shape)

    def encode_text(
        self,
        class_names: List[str],
        dataset_name: Optional[str] = None,
        use_ensemble: bool = False,
        with_grad: bool = False,
        display_names: bool = False,
    ) -> torch.Tensor:
        """
        用類別名稱生成文字特徵。
        支援標準單一 Prompt 模式與領域多模板集成 (Domain Multi-Prompt Ensemble) 模式。

        Args:
            class_names: 類別名稱列表，長度 C
            dataset_name: 資料集名稱 (可選, 用於調用專屬領域 Prompt)
            use_ensemble: 若為 True，則啟用領域專屬多模板特徵平均集成
            with_grad: 文字塔有 LoRA 時需為 True，讓梯度流回文字端 LoRA
            display_names: 預設路徑下依 dataset_name 查 DISPLAY_NAMES 換成完整名稱
                （預設 False，維持舊實驗的「a photo of a MEL」行為）

        Returns:
            text_feat: [C, d] L2 normalized 文字特徵
        """
        if use_ensemble and dataset_name:
            if with_grad:
                raise NotImplementedError("Prompt ensemble 尚未支援文字塔 LoRA（ensemble 路徑固定在 no_grad 下計算）")
            from utils.prompt_templates import get_domain_text_embeddings
            return get_domain_text_embeddings(
                clip_model=self.model,
                class_names=class_names,
                dataset_name=dataset_name,
                device=self.device,
                use_ensemble=True,
            )

        # 預設通用 Prompt 模式
        if display_names:
            class_names = [display_name(n, dataset_name) for n in class_names]
        prompts = [f"a photo of a {name.replace('___', ' ').replace('__', ' ').replace('_', ' ')}" for name in class_names]
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            text_feat = self.model.encode_text(tokens)   # [C, d]

        text_feat = F.normalize(text_feat.float(), dim=-1)
        return text_feat

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """僅提取 CLS 特徵（用於推論）"""
        cls_feat, _ = self.encode_image_with_patches(images)
        return cls_feat

    def logit_scale(self) -> float:
        return self.model.logit_scale.exp().item()
