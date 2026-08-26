"""
experiments/sr_pipeline_study.py
Real-ESRGAN + CC-CDFSL 整合實驗（Phase 2）

目的：驗證在降解影像上，Pre-trained SR 能否挽回準確率損失
流程：
  原始 (224) → 降解 (56×56) → Real-ESRGAN SR → CLIP 編碼 → CC-CDFSL 評估

比較三種條件：
  A. Original:  原始 224×224 影像
  B. Degraded:  56×56 bicubic 降解（已知掉 ~9%）
  C. SR:        降解後用 Real-ESRGAN 恢復

用法：
    # 下載模型後執行（模型會自動下載）
    python experiments/sr_pipeline_study.py --dataset eurosat --n_shot 5 --n_episodes 50

    # 指定降解等級（預設 56）
    python experiments/sr_pipeline_study.py --dataset isic --degrade_size 56 --n_episodes 50

    # 全部資料集
    python experiments/sr_pipeline_study.py --dataset all --n_shot 5 --n_episodes 100
"""
import sys
import os
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 把專案根目錄和 Real-ESRGAN 加入 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REALESRGAN_ROOT = PROJECT_ROOT / "Real-ESRGAN"
sys.path.insert(0, str(REALESRGAN_ROOT))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform
from utils.metrics import compute_few_shot_accuracy, compute_confidence_interval

# Real-ESRGAN imports（需要在 Real-ESRGAN 目錄下才能正確 import）
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# CLIP normalization 統計數值（用於 tensor ↔ numpy 轉換）
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# ─────────────────────────────────────────────
# SR 相關工具函數
# ─────────────────────────────────────────────

def build_realesrgan(device: str, model_name: str = "RealESRGAN_x4plus") -> RealESRGANer:
    """
    初始化 Real-ESRGAN upsampler。

    使用 RealESRGAN_x4plus（x4 scale），搭配 outscale=1 來做
    「低解析度 → 4×SR → resize 回 224」的還原流程。

    Args:
        device:     'cuda' or 'cpu'
        model_name: Real-ESRGAN 模型名稱

    Returns:
        RealESRGANer 實例
    """
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=4
    )

    # 模型 weight 路徑（若不存在會自動下載）
    weight_path = REALESRGAN_ROOT / "weights" / f"{model_name}.pth"
    if not weight_path.exists():
        # 自動下載
        model_url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
        logger.info(f"Weight not found. Will auto-download from: {model_url}")
        model_path_str = model_url
    else:
        model_path_str = str(weight_path)
        logger.info(f"Using cached weights: {weight_path}")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path_str,
        model=model,
        tile=0,          # 不分 tile（影像夠小）
        tile_pad=10,
        pre_pad=0,
        half=True,       # fp16 加速（RTX 3060/4090 都支援）
        device=torch.device(device),
    )

    logger.info(f"Real-ESRGAN ready on {device}")
    return upsampler


def clip_tensor_to_bgr_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    將 CLIP normalized tensor 轉回 BGR uint8 numpy（Real-ESRGAN 的輸入格式）。

    Args:
        tensor: [3, H, W] CLIP normalized tensor

    Returns:
        [H, W, 3] BGR uint8 numpy array
    """
    # Step 1: Denormalize
    mean = CLIP_MEAN.to(tensor.device).view(3, 1, 1)
    std  = CLIP_STD.to(tensor.device).view(3, 1, 1)
    img = tensor * std + mean  # [3, H, W], range [0, 1]

    # Step 2: Clamp 並轉換為 uint8
    img = img.clamp(0, 1).cpu().numpy()              # [3, H, W]
    img = (img * 255).round().astype(np.uint8)       # uint8
    img = np.transpose(img, (1, 2, 0))              # [H, W, 3] RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # BGR（Real-ESRGAN 格式）
    return img_bgr


def bgr_numpy_to_clip_tensor(
    bgr: np.ndarray,
    target_size: int = 224,
    device: str = "cuda",
) -> torch.Tensor:
    """
    將 Real-ESRGAN 輸出的 BGR uint8 numpy 轉為 CLIP normalized tensor。

    Args:
        bgr:         [H, W, 3] BGR uint8
        target_size: CLIP 輸入尺寸（224）
        device:      目標設備

    Returns:
        [3, target_size, target_size] CLIP normalized tensor
    """
    # Step 1: BGR → RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3] uint8

    # Step 2: Resize 到 CLIP 輸入大小（用 LANCZOS4 最高品質）
    rgb_resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Step 3: 轉 float tensor 並 normalize
    img = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0  # [3, H, W]
    mean = CLIP_MEAN.view(3, 1, 1)
    std  = CLIP_STD.view(3, 1, 1)
    img = (img - mean) / std

    return img.to(device)


def degrade_tensor_batch(images: torch.Tensor, degrade_size: int) -> torch.Tensor:
    """
    對 tensor batch 做 bicubic 降解（downscale → upscale 回原始尺寸）。

    Args:
        images:       [B, 3, H, W]
        degrade_size: 降解目標尺寸

    Returns:
        degraded: [B, 3, H, W]
    """
    target_size = images.shape[-1]
    down = F.interpolate(images, size=(degrade_size, degrade_size),
                         mode="bicubic", align_corners=False)
    up   = F.interpolate(down, size=(target_size, target_size),
                         mode="bicubic", align_corners=False)
    return up


@torch.no_grad()
def sr_enhance_batch(
    images: torch.Tensor,         # [B, 3, 224, 224] CLIP tensor（已降解）
    upsampler: RealESRGANer,
    device: str,
    target_size: int = 224,
) -> torch.Tensor:
    """
    對整個 batch 的 CLIP tensor 做 Real-ESRGAN SR，返回 CLIP tensor。

    流程：
        CLIP normalized tensor
        → denormalize → BGR uint8 numpy
        → RealESRGANer.enhance（4× upscale）
        → resize 回 224 → CLIP normalize
        → CLIP normalized tensor

    Args:
        images:      [B, 3, H, W] CLIP normalized tensor（已降解）
        upsampler:   RealESRGANer 實例
        device:      目標設備
        target_size: 最終尺寸（CLIP 的 224）

    Returns:
        sr_images:   [B, 3, target_size, target_size] SR 後的 CLIP tensor
    """
    sr_tensors = []
    for img_tensor in images:
        # 1. CLIP tensor → BGR numpy
        bgr = clip_tensor_to_bgr_numpy(img_tensor)

        # 2. Real-ESRGAN enhance（4× upscale）
        try:
            sr_bgr, _ = upsampler.enhance(bgr, outscale=4)
        except RuntimeError as e:
            # CUDA OOM 等錯誤：fallback 到 bicubic
            logger.warning(f"Real-ESRGAN enhance failed: {e}. Falling back to bicubic.")
            sr_bgr = bgr

        # 3. BGR numpy → CLIP tensor（含 resize 回 224）
        sr_tensor = bgr_numpy_to_clip_tensor(sr_bgr, target_size=target_size, device=device)
        sr_tensors.append(sr_tensor)

    return torch.stack(sr_tensors)  # [B, 3, 224, 224]


# ─────────────────────────────────────────────
# 評估函數
# ─────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    clip_lora: CLIPLoRA,
    sampler: FewShotEpisodeSampler,
    n_episodes: int,
    device: str,
    mode: str,                            # "original" | "degraded" | "sr"
    degrade_size: Optional[int] = None,
    upsampler: Optional[RealESRGANer] = None,
) -> Dict[str, float]:
    """
    三種模式的 few-shot 評估。

    Args:
        mode:         "original" / "degraded" / "sr"
        degrade_size: 降解目標尺寸（mode != "original" 時必填）
        upsampler:    RealESRGANer（mode == "sr" 時必填）

    Returns:
        {"mean_acc": float, "ci": float}
    """
    clip_lora.eval()
    accs = []

    for ep in range(n_episodes):
        episode = sampler.sample()

        query_images = episode["query_images"].to(device)   # [K*Q, 3, 224, 224]
        query_labels = episode["query_labels"].to(device)
        class_names  = episode["class_names"]

        # 依照模式處理 query 影像
        if mode == "degraded":
            query_images = degrade_tensor_batch(query_images, degrade_size)

        elif mode == "sr":
            # 先降解，再 SR 還原
            degraded = degrade_tensor_batch(query_images, degrade_size)
            query_images = sr_enhance_batch(degraded, upsampler, device)

        # 提取文字和影像特徵
        text_feat = clip_lora.encode_text(class_names)
        q_cls, _  = clip_lora.encode_image_with_patches(query_images)
        q_logits  = q_cls @ text_feat.T

        acc = compute_few_shot_accuracy(q_logits, query_labels)
        accs.append(acc)

        if (ep + 1) % 20 == 0 or (ep + 1) == n_episodes:
            mean_tmp, ci_tmp = compute_confidence_interval(accs)
            logger.info(f"  [{mode}] Episode {ep+1}/{n_episodes}: {mean_tmp:.2f} ± {ci_tmp:.2f}%")

    mean_acc, ci = compute_confidence_interval(accs)
    return {"mean_acc": mean_acc, "ci": ci}


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Real-ESRGAN + CC-CDFSL 整合評估"
    )
    parser.add_argument("--dataset", type=str, default="eurosat",
                        choices=["eurosat", "isic", "chestx", "crop_disease", "all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=100)

    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--degrade_size", type=int, default=56,
                        help="降解目標尺寸（建議用 56，Phase 1 結果最差的等級）")
    parser.add_argument("--skip_original", action="store_true",
                        help="跳過 Original 評估（已有 Phase 1 結果時可省時間）")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None)

    return parser.parse_args()


def run_dataset_study(dataset_name: str, args, device: str, upsampler: RealESRGANer) -> List[Dict]:
    """對單一資料集執行完整的三條件比較實驗"""
    logger.info(f"\n{'='*60}")
    logger.info(f"SR Pipeline Study: {dataset_name} ({args.n_shot}-shot, degrade={args.degrade_size})")
    logger.info(f"{'='*60}")

    # 資料集與 sampler
    transform = get_clip_transform(img_size=224)
    dataset = get_dataset(dataset_name, args.data_root, split="test", transform=transform)
    sampler = FewShotEpisodeSampler(
        dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query
    )
    logger.info(f"Dataset: {dataset}")

    # 模型
    clip_wrapper = CLIPWrapper(backbone=args.backbone, device=device)
    clip_lora    = CLIPLoRA(clip_wrapper=clip_wrapper, r=args.lora_r, alpha=args.lora_alpha)
    clip_lora.eval()

    # 載入 checkpoint
    ckpt_path = Path(args.checkpoint_dir) / f"{dataset_name}_{args.n_shot}shot_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        key  = "lora_state_dict" if "lora_state_dict" in ckpt else None
        clip_lora.load_state_dict(ckpt[key] if key else ckpt, strict=False)
        logger.info(f"Loaded: {ckpt_path}")
    else:
        logger.warning("No checkpoint found, using zero-shot CLIP")

    results = []

    # ── A. Original ──────────────────────────────────────────
    if not args.skip_original:
        logger.info(f"\n[A] Original (224×224)...")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        r = run_evaluation(clip_lora, sampler, args.n_episodes, device, mode="original")
        r.update({"dataset": dataset_name, "mode": "A_original", "degrade_size": "224"})
        results.append(r)
    else:
        logger.info("[A] Skipping original evaluation (--skip_original)")

    # ── B. Degraded ───────────────────────────────────────────
    logger.info(f"\n[B] Degraded ({args.degrade_size}×{args.degrade_size})...")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = run_evaluation(clip_lora, sampler, args.n_episodes, device,
                       mode="degraded", degrade_size=args.degrade_size)
    r.update({"dataset": dataset_name, "mode": "B_degraded",
              "degrade_size": str(args.degrade_size)})
    results.append(r)

    # ── C. SR ─────────────────────────────────────────────────
    logger.info(f"\n[C] SR (degrade {args.degrade_size} → Real-ESRGAN → 224)...")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    r = run_evaluation(clip_lora, sampler, args.n_episodes, device,
                       mode="sr", degrade_size=args.degrade_size, upsampler=upsampler)
    r.update({"dataset": dataset_name, "mode": "C_sr",
              "degrade_size": str(args.degrade_size)})
    results.append(r)

    return results


def print_summary_table(all_results: List[Dict], n_shot: int, degrade_size: int):
    """列印最終彙整表格"""
    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2 RESULTS  ({n_shot}-shot, degrade={degrade_size}×{degrade_size}→SR)")
    logger.info("=" * 80)
    logger.info(f"{'Dataset':<16} {'Original(224)':<18} {'Degraded':<18} {'SR Restored':<18} {'SR vs Orig':>10} {'SR vs Deg':>10}")
    logger.info("-" * 90)

    datasets = []
    for r in all_results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    all_orig, all_deg, all_sr = [], [], []

    for ds in datasets:
        ds_results = {r["mode"]: r for r in all_results if r["dataset"] == ds}
        orig = ds_results.get("A_original", {}).get("mean_acc", None)
        deg  = ds_results.get("B_degraded", {}).get("mean_acc")
        sr   = ds_results.get("C_sr", {}).get("mean_acc")

        orig_str = f"{orig:.2f}%" if orig is not None else "skipped"
        deg_str  = f"{deg:.2f}%"
        sr_str   = f"{sr:.2f}%"

        sr_vs_orig = f"{sr - orig:+.2f}%" if orig is not None else "N/A"
        sr_vs_deg  = f"{sr - deg:+.2f}%"

        logger.info(f"{ds:<16} {orig_str:<18} {deg_str:<18} {sr_str:<18} {sr_vs_orig:>10} {sr_vs_deg:>10}")

        if orig is not None: all_orig.append(orig)
        all_deg.append(deg)
        all_sr.append(sr)

    logger.info("-" * 90)

    avg_str = f"{np.mean(all_orig):.2f}%" if all_orig else "skipped"
    avg_deg = f"{np.mean(all_deg):.2f}%"
    avg_sr  = f"{np.mean(all_sr):.2f}%"
    avg_vs_orig = f"{np.mean(all_sr) - np.mean(all_orig):+.2f}%" if all_orig else "N/A"
    avg_vs_deg  = f"{np.mean(all_sr) - np.mean(all_deg):+.2f}%"

    logger.info(f"{'Average':<16} {avg_str:<18} {avg_deg:<18} {avg_sr:<18} {avg_vs_orig:>10} {avg_vs_deg:>10}")

    logger.info("\n📊 解讀提示：")
    logger.info("  SR vs Deg > 0  → SR 確實幫助還原了準確率")
    logger.info("  SR vs Orig ≈ 0 → SR 幾乎完全彌補了降解損失（理想情況）")
    logger.info("  SR vs Orig < 0 → SR 有幫助但仍有差距（仍有研究空間 = 你的論文貢獻）")


def save_csv(all_results: List[Dict], output_path: str):
    import csv
    fieldnames = ["dataset", "mode", "degrade_size", "mean_acc", "ci"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"Results saved to: {output_path}")


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 初始化 Real-ESRGAN（整個實驗只載入一次）──
    logger.info("Loading Real-ESRGAN model...")
    upsampler = build_realesrgan(device=device)
    logger.info("Real-ESRGAN loaded successfully!")

    # 資料集列表
    datasets = ["isic", "chestx", "eurosat", "crop_disease"] if args.dataset == "all" else [args.dataset]

    # 執行實驗
    all_results = []
    for dataset_name in datasets:
        results = run_dataset_study(dataset_name, args, device, upsampler)
        all_results.extend(results)

    # 輸出結果
    print_summary_table(all_results, args.n_shot, args.degrade_size)

    csv_path = args.output_csv or f"experiments/sr_pipeline_results_{args.n_shot}shot.csv"
    save_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
