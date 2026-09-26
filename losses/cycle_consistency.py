"""
losses/cycle_consistency.py
CC-CDFSL 循環一致性損失

實作論文 Section 3.2 的兩個 cycle consistency 損失：
  1. Text-to-Image-to-Text (T-I-T) Cycle Loss  (Eq. 4-9)
  2. Image-to-Text-to-Image (I-T-I) Cycle Loss (Eq. 10-15)
  3. Semantic Anchor Module (Augmentation + Shrinking)
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicConsistencyLoss(nn.Module):
    """
    CC-CDFSL 核心損失模組。

    負責計算：
    - T-I-T cycle consistency loss (L_cyc_txt)
    - I-T-I cycle consistency loss (L_cyc_img)

    Args:
        top_k: Semantic Anchor Shrinking 選取的 patch 數（論文設 k=10）
    """

    def __init__(self, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

    # ---------------------------------------------------------------
    # T-I-T Cycle Consistency（Eq. 4-9）
    # ---------------------------------------------------------------

    def compute_tit_loss(
        self,
        text_feat: torch.Tensor,
        patch_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Text-to-Image-to-Text Cycle Consistency Loss。

        流程：
          text_feat (T_j) → 找最相似 patch (L*_j) → 計算 E_txt → 重建 T^rec_j
          Loss = 1 - mean cosine_sim(T_j, T^rec_j)

        Args:
            text_feat:  [C, d]   L2-normalized 文字特徵（C = 類別數）
            patch_feat: [N, d]   L2-normalized patch 特徵（N = 所有 support 圖像的 patches）

        Returns:
            L_cyc_txt: scalar tensor
        """
        # text_feat: [C, d], patch_feat: [N, d]
        # 所有特徵應已 L2 normalized

        C = text_feat.shape[0]

        # Eq. 4-5: D_txt[j, i] = T_j · L_i，即文字-patch 相似度矩陣
        # shape: [C, N]
        D_txt = text_feat @ patch_feat.T            # [C, N]

        # Eq. 7: 對每個文字特徵選最相似 patch
        # L*_j = L[argmax_i D_txt[j, i]]
        best_patch_idx = D_txt.argmax(dim=1)        # [C]
        L_star = patch_feat[best_patch_idx]          # [C, d]

        # Eq. 8: E_txt = L* · T^T ∈ [C, C]
        E_txt = L_star @ text_feat.T                 # [C, C]

        # Eq. 9: 重建文字特徵
        # T^rec_j = T[argmax_k E_txt[j, k]]
        best_text_idx = E_txt.argmax(dim=1)          # [C]
        T_rec = text_feat[best_text_idx]             # [C, d]

        # Cycle consistency loss
        # sim(T_j, T^rec_j) = dot product（因為都已 L2 normalized）
        sim = (text_feat * T_rec).sum(dim=-1)        # [C]
        loss = 1.0 - sim.mean()

        return loss

    # ---------------------------------------------------------------
    # Semantic Anchor Module - Shrinking (Eq. 10-12)
    # ---------------------------------------------------------------

    def select_anchor_features(
        self,
        patch_feat_per_image: torch.Tensor,   # [B*(A+1), P, d]
        text_feat: torch.Tensor,              # [C, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Semantic Anchor Module - Shrinking Phase（Eq. 10-12）

        對每個影像的每個類別，選 top-k 相似 patches，合併去重後得到 anchor 特徵。

        Args:
            patch_feat_per_image: [B*(A+1), P, d] 每個影像的 patch 特徵
                B = 原始影像數，A+1 = augmentation 後（含原始）
            text_feat: [C, d] 文字特徵

        Returns:
            X_anchor: [V, d] anchor patch 特徵
            anchor_indices: [V] anchor 的全域索引（在展平後的 patch 矩陣中）
        """
        B_aug, P, d = patch_feat_per_image.shape
        C = text_feat.shape[0]
        k = self.top_k

        # 展平成 [B_aug * P, d]
        all_patches = patch_feat_per_image.reshape(B_aug * P, d)   # [N_total, d]

        # Eq. 6/10: D_txt[b, j, i] = T_j · L^(b)_i
        # 為節省記憶體，逐影像計算
        # D_txt_per_img: [B_aug, C, P]
        D_txt_per_img = torch.einsum(
            "bpd,cd->bcp",
            patch_feat_per_image,   # [B_aug, P, d]
            text_feat               # [C, d]
        )                           # [B_aug, C, P]

        # Eq. 10: I^(b)_j = top-k_i(D_txt[b, j, i])
        # 得到每個（影像, 類別）的 top-k patch 局部索引
        # topk_local_idx: [B_aug, C, k]
        _, topk_local_idx = D_txt_per_img.topk(k=min(k, P), dim=-1)   # [B_aug, C, k]

        # 轉成全域索引（在 all_patches 中的位置）
        # 第 b 個影像的第 i 個 patch 全域索引 = b * P + i
        offsets = torch.arange(B_aug, device=patch_feat_per_image.device).view(B_aug, 1, 1) * P
        global_idx = (topk_local_idx + offsets).reshape(-1)   # [B_aug * C * k]

        # Eq. 11: 去重
        anchor_indices = torch.unique(global_idx)              # [V]

        # Eq. 12: X_anchor = L[I_anchor]
        X_anchor = all_patches[anchor_indices]                 # [V, d]

        return X_anchor, anchor_indices

    # ---------------------------------------------------------------
    # I-T-I Cycle Consistency（Eq. 13-15）
    # ---------------------------------------------------------------

    def compute_iti_loss(
        self,
        X_anchor: torch.Tensor,   # [V, d] anchor patch 特徵（來自原始影像）
        X_aug: torch.Tensor,      # [N_aug, d] augmented patch 特徵（展平）
        text_feat: torch.Tensor,  # [C, d]
    ) -> torch.Tensor:
        """
        Image-to-Text-to-Image Cycle Consistency Loss（Eq. 13-15）

        流程：
          anchor patch (x_n) → 最相似文字 (t_n) → 在 augmented space 中找最相似 patch (x_hat_n)
          Loss = 1 - mean cosine_sim(x_n, x_hat_n)

        Args:
            X_anchor: [V, d]    anchor patch 特徵（已 L2 normalized）
            X_aug:    [N_aug, d] augmented 全部 patch（已 L2 normalized）
            text_feat: [C, d]   文字特徵（已 L2 normalized）

        Returns:
            L_cyc_img: scalar tensor
        """
        V = X_anchor.shape[0]

        # Eq. 13: 對每個 anchor patch 找最相似文字
        # sim_xt: [V, C]
        sim_xt = X_anchor @ text_feat.T           # [V, C]
        best_text_idx = sim_xt.argmax(dim=-1)     # [V]
        t_n = text_feat[best_text_idx]            # [V, d]

        # Eq. 14: 用文字特徵在 augmented space 找最相似 patch
        # sim_tx_aug: [V, N_aug]
        sim_tx_aug = t_n @ X_aug.T               # [V, N_aug]
        best_aug_idx = sim_tx_aug.argmax(dim=-1)  # [V]
        x_hat = X_aug[best_aug_idx]              # [V, d]

        # Eq. 15: I-T-I cycle loss
        sim = (X_anchor * x_hat).sum(dim=-1)     # [V]
        loss = 1.0 - sim.mean()

        return loss

    # ---------------------------------------------------------------
    # 完整 forward：同時計算 T-I-T 和 I-T-I loss
    # ---------------------------------------------------------------

    def forward(
        self,
        text_feat: torch.Tensor,             # [C, d]
        patch_feat_orig: torch.Tensor,       # [B, P, d] 原始影像 patches
        patch_feat_aug: Optional[torch.Tensor] = None,  # [B*A, P, d] augmented patches
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算完整的 CC-CDFSL 循環一致性損失。

        Args:
            text_feat:      [C, d]      L2-normalized 文字特徵
            patch_feat_orig: [B, P, d]  原始圖像 patch 特徵
            patch_feat_aug:  [B*A, P, d] augmented 圖像 patch 特徵（若為 None 則跳過 I-T-I）

        Returns:
            L_cyc_txt: T-I-T cycle loss
            L_cyc_img: I-T-I cycle loss（若無 aug 則為 0）
        """
        B, P, d = patch_feat_orig.shape

        # ---------- T-I-T Loss ----------
        # 將所有影像的 patches 展平（包含 augmented patches 以擴大語料庫）
        if patch_feat_aug is not None:
            all_patches_for_tit = torch.cat([
                patch_feat_orig.reshape(-1, d),    # [B*P, d]
                patch_feat_aug.reshape(-1, d),     # [B*A*P, d]
            ], dim=0)                              # [(B+B*A)*P, d]
        else:
            all_patches_for_tit = patch_feat_orig.reshape(-1, d)  # [B*P, d]

        L_cyc_txt = self.compute_tit_loss(text_feat, all_patches_for_tit)

        # ---------- I-T-I Loss ----------
        if patch_feat_aug is not None:
            # 合併原始 + augmented 進行 Shrinking
            all_patches_per_img = torch.cat([
                patch_feat_orig,    # [B, P, d]
                patch_feat_aug,     # [B*A, P, d]
            ], dim=0)               # [B*(A+1), P, d]

            X_anchor, _ = self.select_anchor_features(all_patches_per_img, text_feat)

            # augmented patches 展平（作為搜索空間）
            X_aug_flat = patch_feat_aug.reshape(-1, d)  # [B*A*P, d]

            L_cyc_img = self.compute_iti_loss(X_anchor, X_aug_flat, text_feat)
        else:
            L_cyc_img = torch.tensor(0.0, device=text_feat.device)

        return L_cyc_txt, L_cyc_img


class PaperCyclicConsistencyLoss(nn.Module):
    """
    照 CC-CDFSL 論文 Eq. 3–15 字面實作（官方 repo github.com/z-yaz/CC-CDFSL 只有 README，未釋出程式碼）。

    與舊版 CyclicConsistencyLoss 的差異：
      1. Eq. 5：patch 特徵先經可訓練 2 層 MLP（ReLU(L'W1)W2，無 bias）再 L2 normalize，
         T-I-T、Shrinking、I-T-I 全部在 MLP 空間進行。
      2. Eq. 14：I-T-I 只在 anchor 所屬影像的其他視角中檢索。論文寫「x_n 的增強空間」，但 anchor 本身
         也在該空間內，照字面常會檢索到自己使 loss 恆為 0，因此排除 anchor 自己所在的視角。
      3. T-I-T 照 Eq. 7–9：兩次 argmax 後 loss 只由文字特徵組成，影像端（含 MLP）拿不到梯度，
         只有文字塔 LoRA 會被更新。

    MLP 只從 I-T-I 拿到梯度，而 I-T-I 可被常數輸出滿足 ⇒ 有坍縮風險。
    forward 額外回傳 mlp_cos（MLP 輸出隨機 patch 兩兩平均餘弦），逼近 1 代表坍縮。
    """

    def __init__(self, dim: int, top_k: int = 10):
        super().__init__()
        self.top_k = top_k
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(
        self,
        text_feat: torch.Tensor,        # [C, d]
        patch_feat_orig: torch.Tensor,  # [B, P, d]
        patch_feat_aug: torch.Tensor,   # [A*B, P, d]，排列為 a*B + b
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, d = patch_feat_orig.shape
        A = patch_feat_aug.shape[0] // B
        n_views = A + 1
        T = text_feat

        views = torch.cat([
            patch_feat_orig.unsqueeze(1),
            patch_feat_aug.view(A, B, P, d).transpose(0, 1),
        ], dim=1)                                           # [B, A+1, P, d]，視角 0 為原圖
        L = F.normalize(self.mlp(views), dim=-1)            # Eq. 5
        flat = L.reshape(-1, d)                             # [H, d]，H = B(A+1)P

        # T-I-T（Eq. 6–9）
        L_star = flat[(T @ flat.T).argmax(dim=1)]          # [C, d]
        T_rec = T[(L_star @ T.T).argmax(dim=1)]            # [C, d]
        loss_tit = 1.0 - (T * T_rec).sum(dim=-1).mean()

        # Shrinking（Eq. 10–12）：每張影像、每個視角、每個類別取 top-k
        k = min(self.top_k, P)
        topk = torch.einsum("bvpd,cd->bvcp", L, T).topk(k, dim=-1).indices          # [B, V, C, k]
        view_offset = torch.arange(n_views, device=L.device).view(1, n_views, 1, 1) * P
        local_idx = (topk + view_offset).reshape(B, -1)                               # 影像內索引 v*P + p
        per_image = L.reshape(B, n_views * P, d)
        cand_view = torch.arange(n_views * P, device=L.device) // P

        # I-T-I（Eq. 13–15）
        sims = []
        for b in range(B):
            idx = torch.unique(local_idx[b])
            x = per_image[b, idx]                                        # anchors [n, d]
            t = T[(x @ T.T).argmax(dim=-1)]                              # Eq. 13
            s = t @ per_image[b].T                                       # [n, V*P]
            s = s.masked_fill((idx // P)[:, None] == cand_view[None, :], float("-inf"))
            x_hat = per_image[b, s.argmax(dim=-1)]                       # Eq. 14
            sims.append((x * x_hat).sum(dim=-1))
        loss_iti = 1.0 - torch.cat(sims).mean()                          # Eq. 15

        with torch.no_grad():
            sample = flat[torch.randint(0, flat.shape[0], (256,), device=flat.device)].float()
            gram = sample @ sample.T
            mlp_cos = (gram.sum() - gram.diagonal().sum()) / (256 * 255)

        return loss_tit, loss_iti, mlp_cos
