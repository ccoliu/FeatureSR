"""
models/clip_lora.py
CLIP-LoRA：在 CLIP ViT 的 attention projection 上套用 Low-Rank Adaptation。

基於論文設定：
- 套用在 visual encoder 的 Q, K, V, Out projection
- rank r = 4, alpha = 1
- 其他參數凍結
"""
import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA 線性層，公式：W' = W + (B @ A) * (alpha / r)

    Args:
        in_features: 輸入維度
        out_features: 輸出維度
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: dropout rate
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 原始凍結權重（只是一個佔位，實際由 patch_lora_to_model 注入）
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_param = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        # LoRA 可訓練參數
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始線性層
        result = F.linear(x, self.weight, self.bias_param)
        # LoRA 增量
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"


class CLIPLoRA(nn.Module):
    """
    將 LoRA 套用到 CLIP ViT visual encoder 的 attention 層。

    套用位置：每個 transformer block 的
    - in_proj (Q, K, V combined projection)
    - out_proj

    Args:
        clip_wrapper: CLIPWrapper 實例
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
    """

    def __init__(
        self,
        clip_wrapper,
        r: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.clip = clip_wrapper
        self.r = r
        self.alpha = alpha

        # 注入 LoRA 到 visual transformer 的 attention 層
        self.lora_layers = nn.ModuleList()
        self._inject_lora(r, alpha, dropout)

        # 確保 LoRA 參數與 CLIP 模型在相同 device
        device = clip_wrapper.device
        self.lora_layers.to(device)

    def _inject_lora(self, r: int, alpha: float, dropout: float):
        """將 LoRA 注入 CLIP ViT 的 attention projection"""
        visual = self.clip.model.visual

        # 只有 ViT 有 transformer.resblocks
        if not hasattr(visual, "transformer"):
            raise ValueError("CLIPLoRA currently only supports ViT-based CLIP backbones.")

        for block in visual.transformer.resblocks:
            attn = block.attn

            # ========== in_proj（合併 Q, K, V）==========
            # CLIP 使用 in_proj_weight [3*embed, embed]
            embed_dim = attn.embed_dim
            in_proj_weight = attn.in_proj_weight  # [3*embed, embed]
            in_proj_bias = attn.in_proj_bias       # [3*embed]

            lora_in = LoRALinear(
                in_features=embed_dim,
                out_features=3 * embed_dim,
                r=r, alpha=alpha, dropout=dropout,
                bias=(in_proj_bias is not None),
            )
            lora_in.weight = nn.Parameter(in_proj_weight.detach(), requires_grad=False)
            if in_proj_bias is not None:
                lora_in.bias_param = nn.Parameter(in_proj_bias.detach(), requires_grad=False)

            # ========== out_proj ==========
            out_proj = attn.out_proj
            out_dim, in_dim = out_proj.weight.shape
            lora_out = LoRALinear(
                in_features=in_dim,
                out_features=out_dim,
                r=r, alpha=alpha, dropout=dropout,
                bias=(out_proj.bias is not None),
            )
            lora_out.weight = nn.Parameter(out_proj.weight.detach(), requires_grad=False)
            if out_proj.bias is not None:
                lora_out.bias_param = nn.Parameter(out_proj.bias.detach(), requires_grad=False)

            # 替換
            attn.in_proj_lora = lora_in
            attn.out_proj_lora = lora_out
            self.lora_layers.append(lora_in)
            self.lora_layers.append(lora_out)

            # 覆寫 forward 方法，使用 LoRA 的投影
            self._patch_attention_forward(attn)

    @staticmethod
    def _patch_attention_forward(attn):
        """
        Monkey-patch CLIP MultiheadAttention 的 forward，
        使其使用 LoRA 版本的 in_proj 和 out_proj。
        """
        original_forward = attn.forward

        def lora_forward(query, key, value, key_padding_mask=None,
                         need_weights=True, attn_mask=None):
            # 計算 Q, K, V（使用 LoRA in_proj）
            tgt_len, bsz, embed_dim = query.shape
            in_proj_out = attn.in_proj_lora(query.view(-1, embed_dim))
            in_proj_out = in_proj_out.view(tgt_len, bsz, 3 * embed_dim)
            q, k, v = in_proj_out.chunk(3, dim=-1)

            head_dim = embed_dim // attn.num_heads
            scaling = float(head_dim) ** -0.5
            q = q * scaling

            # Reshape for multi-head attention
            q = q.contiguous().view(tgt_len, bsz * attn.num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * attn.num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * attn.num_heads, head_dim).transpose(0, 1)

            # Attention weights
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if key_padding_mask is not None:
                attn_weights = attn_weights.view(bsz, attn.num_heads, tgt_len, -1)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                attn_weights = attn_weights.view(bsz * attn.num_heads, tgt_len, -1)
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Attention output
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

            # 使用 LoRA out_proj
            attn_output = attn.out_proj_lora(attn_output.view(-1, embed_dim))
            attn_output = attn_output.view(tgt_len, bsz, embed_dim)

            return attn_output, None

        attn.forward = lora_forward

    def trainable_parameters(self) -> List[nn.Parameter]:
        """只返回 LoRA 可訓練參數"""
        return [p for p in self.lora_layers.parameters() if p.requires_grad]

    def encode_image_with_patches(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """代理到 clip_wrapper"""
        return self.clip.encode_image_with_patches(images)

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """代理到 clip_wrapper"""
        return self.clip.encode_text(class_names)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image_with_patches(images)

    def save_lora(self, path: str):
        """只儲存 LoRA 參數"""
        state = {k: v for k, v in self.state_dict().items() if "lora_" in k}
        torch.save(state, path)

    def load_lora(self, path: str):
        """載入 LoRA 參數"""
        state = torch.load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        return missing, unexpected
