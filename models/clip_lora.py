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

    def reset_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class LoRAQKVSeparate(nn.Module):
    """Frozen in_proj [3E, E] plus independent rank-r LoRA on each of q/k/v (CLIP-LoRA style)."""

    def __init__(self, embed_dim: int, r: int, alpha: float, dropout: float,
                 weight: torch.Tensor, bias: Optional[torch.Tensor], enable=("q", "k", "v")):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.bias_param = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None
        self.scaling = alpha / r
        self.enable = tuple(enable)
        self.lora_A = nn.ParameterDict({n: nn.Parameter(torch.empty(r, embed_dim)) for n in self.enable})
        self.lora_B = nn.ParameterDict({n: nn.Parameter(torch.zeros(embed_dim, r)) for n in self.enable})
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_lora()

    def reset_lora(self):
        for n in self.enable:
            nn.init.kaiming_uniform_(self.lora_A[n], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[n])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias_param)
        xd = self.dropout(x)
        deltas = []
        for n in ("q", "k", "v"):
            if n in self.enable:
                deltas.append(xd @ self.lora_A[n].T @ self.lora_B[n].T * self.scaling)
            else:
                deltas.append(torch.zeros(*x.shape[:-1], self.embed_dim, device=x.device, dtype=out.dtype))
        return out + torch.cat(deltas, dim=-1)


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
        encoder: str = "vision",
        qkv_mode: str = "fused",
        use_out_proj: bool = True,
    ):
        """
        encoder:      "vision" | "text" | "both"
        qkv_mode:     "fused"（單一 rank-r LoRA 作用在合併的 in_proj，舊版預設）
                      | "separate"（q/k/v 各一組 rank-r LoRA，CLIP-LoRA 原論文設定）
        use_out_proj: 是否也在 out_proj 加 LoRA（CLIP-LoRA 原論文預設不加）
        """
        super().__init__()
        self.clip = clip_wrapper
        self.r = r
        self.alpha = alpha
        self.encoder = encoder
        self.text_lora = encoder in ("text", "both")

        self.lora_layers = nn.ModuleList()
        towers = []
        if encoder in ("vision", "both"):
            visual = self.clip.model.visual
            if not hasattr(visual, "transformer"):
                raise ValueError("CLIPLoRA currently only supports ViT-based CLIP backbones.")
            towers.append(visual.transformer)
        if self.text_lora:
            towers.append(self.clip.model.transformer)
        for tower in towers:
            self._inject_lora(tower, r, alpha, dropout, qkv_mode, use_out_proj)

        self.lora_layers.to(clip_wrapper.device)

    def _inject_lora(self, transformer, r: int, alpha: float, dropout: float,
                     qkv_mode: str, use_out_proj: bool):
        """將 LoRA 注入一個 CLIP transformer（視覺或文字）的所有 attention projection"""
        for block in transformer.resblocks:
            attn = block.attn
            embed_dim = attn.embed_dim
            in_proj_weight = attn.in_proj_weight  # [3*embed, embed]
            in_proj_bias = attn.in_proj_bias       # [3*embed]

            if qkv_mode == "separate":
                lora_in = LoRAQKVSeparate(embed_dim, r, alpha, dropout, in_proj_weight, in_proj_bias)
            else:
                lora_in = LoRALinear(
                    in_features=embed_dim,
                    out_features=3 * embed_dim,
                    r=r, alpha=alpha, dropout=dropout,
                    bias=(in_proj_bias is not None),
                )
                lora_in.weight = nn.Parameter(in_proj_weight.detach(), requires_grad=False)
                if in_proj_bias is not None:
                    lora_in.bias_param = nn.Parameter(in_proj_bias.detach(), requires_grad=False)
            attn.in_proj_lora = lora_in
            self.lora_layers.append(lora_in)

            if use_out_proj:
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
                attn.out_proj_lora = lora_out
                self.lora_layers.append(lora_out)

            self._patch_attention_forward(attn)

    def reset_lora(self):
        """把所有 LoRA 參數重新初始化（A: kaiming, B: 0），等同回到預訓練 CLIP。"""
        for layer in self.lora_layers:
            layer.reset_lora()

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

            out_layer = getattr(attn, "out_proj_lora", None)
            if out_layer is not None:
                attn_output = out_layer(attn_output.view(-1, embed_dim))
            else:
                attn_output = F.linear(attn_output.view(-1, embed_dim), attn.out_proj.weight, attn.out_proj.bias)
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

    def encode_text(
        self,
        class_names: List[str],
        dataset_name: Optional[str] = None,
        use_ensemble: bool = False,
    ) -> torch.Tensor:
        """代理到 clip_wrapper；文字塔有 LoRA 時保留梯度"""
        return self.clip.encode_text(
            class_names, dataset_name=dataset_name, use_ensemble=use_ensemble,
            with_grad=self.text_lora,
        )

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
