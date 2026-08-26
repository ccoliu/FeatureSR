"""
experiments/sr_degradation_study.py
超解析度降解驗證實驗

目的：驗證影像解析度對 CC-CDFSL 效能的影響
流程：
  1. 對測試影像模擬不同程度的解析度降解
     (bicubic downscale → bicubic upscale 回 224×224)
  2. 用訓練好的 CC-CDFSL checkpoint 評估降解影像
  3. 輸出比較表格：原始 vs 各降解等級

用法：
    # 評估所有資料集、所有降解等級（5-shot）
    python experiments/sr_degradation_study.py \
        --dataset all --n_shot 5 --n_episodes 100

    # 單一資料集快速測試
    python experiments/sr_degradation_study.py \
        --dataset eurosat --n_shot 5 --n_episodes 50

    # 自訂降解等級
    python experiments/sr_degradation_study.py \
        --dataset eurosat --degrade_sizes 28 56 112
"""
import sys
import os
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as TF_func
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform
from utils.metrics import compute_few_shot_accuracy, compute_confidence_interval

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 影像降解函數
# ─────────────────────────────────────────────

def degrade_images(
    images: torch.Tensor,
    degrade_size: int,
    target_size: int = 224,
) -> torch.Tensor:
    """
    模擬解析度降解：bicubic downscale → bicubic upscale。

    這模擬了實際場景中低解析度影像被 resize 到 CLIP 輸入尺寸的情況。

    Args:
        images:       [B, 3, H, W] 原始影像 tensor（已正規化）
        degrade_size: 降解目標尺寸（例如 56, 112）
        target_size:  最終尺寸（CLIP 的 224）

    Returns:
        degraded:     [B, 3, target_size, target_size] 降解後影像
    """
    if degrade_size >= target_size:
        return images  # 不降解

    # Step 1: Downscale（模擬低解析度拍攝）
    downscaled = TF_func.interpolate(
        images,
        size=(degrade_size, degrade_size),
        mode="bicubic",
        align_corners=False,
    )

    # Step 2: Upscale 回原始尺寸（模擬 naïve resize）
    upscaled = TF_func.interpolate(
        downscaled,
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
    )

    return upscaled


def compute_psnr(original: torch.Tensor, degraded: torch.Tensor) -> float:
    """
    計算 PSNR（Peak Signal-to-Noise Ratio）作為降解程度的參考指標。

    注意：因為 CLIP 使用了正規化，這裡的 PSNR 是在正規化空間計算的，
    主要用於相對比較，而非絕對品質。

    Args:
        original: [B, 3, H, W]
        degraded: [B, 3, H, W]

    Returns:
        mean PSNR (dB)
    """
    mse = (original - degraded).pow(2).mean(dim=[1, 2, 3])  # [B]
    # 避免除以零
    mse = mse.clamp(min=1e-10)
    # 在正規化空間中，pixel range 大約在 [-2, 3]，這裡用 max 值
    max_val = max(original.max().item(), 3.0)
    psnr = 10.0 * torch.log10(max_val**2 / mse)  # [B]
    return psnr.mean().item()


# ─────────────────────────────────────────────
# 評估函數
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_degradation(
    clip_lora: CLIPLoRA,
    sampler: FewShotEpisodeSampler,
    n_episodes: int,
    device: str,
    degrade_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    在可選的降解條件下進行 few-shot 評估。

    Args:
        clip_lora:    CLIPLoRA 模型
        sampler:      FewShotEpisodeSampler
        n_episodes:   評估 episode 數
        device:       設備
        degrade_size: 降解尺寸（None = 不降解）

    Returns:
        {
            "mean_acc": float,
            "ci": float,
            "mean_psnr": float,  # 若有降解
        }
    """
    clip_lora.eval()
    accs = []
    psnrs = []

    for ep in range(n_episodes):
        episode = sampler.sample()

        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        class_names = episode["class_names"]

        # 對 query 和 support 都做降解（模擬整體低解析度場景）
        if degrade_size is not None:
            original_query = query_images.clone()
            query_images = degrade_images(query_images, degrade_size)
            psnr_val = compute_psnr(original_query, query_images)
            psnrs.append(psnr_val)

        # 提取文字特徵
        text_feat = clip_lora.encode_text(class_names)

        # 提取 query 特徵
        q_cls, _ = clip_lora.encode_image_with_patches(query_images)
        q_logits = q_cls @ text_feat.T

        acc = compute_few_shot_accuracy(q_logits, query_labels)
        accs.append(acc)

        if (ep + 1) % 20 == 0:
            mean_tmp, ci_tmp = compute_confidence_interval(accs)
            deg_str = f"degrade={degrade_size}" if degrade_size else "original"
            logger.info(
                f"  [{deg_str}] Episode {ep+1}/{n_episodes}: "
                f"{mean_tmp:.2f} ± {ci_tmp:.2f}%"
            )

    mean_acc, ci = compute_confidence_interval(accs)
    result = {"mean_acc": mean_acc, "ci": ci}

    if psnrs:
        result["mean_psnr"] = np.mean(psnrs)

    return result


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CC-CDFSL 降解驗證實驗：量化解析度對效能的影響"
    )

    parser.add_argument("--dataset", type=str, default="eurosat",
                        choices=["eurosat", "isic", "chestx", "crop_disease", "all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="每個配置的評估 episode 數")

    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--degrade_sizes", type=int, nargs="+",
                        default=[56, 112],
                        help="降解目標尺寸列表（預設：56, 112）")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="可選的 CSV 輸出路徑")

    return parser.parse_args()


def run_degradation_study(
    dataset_name: str,
    args,
    device: str,
) -> List[Dict]:
    """對單一資料集進行完整降解實驗"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Degradation Study: {dataset_name} ({args.n_shot}-shot)")
    logger.info(f"Degrade sizes: {args.degrade_sizes}")
    logger.info(f"{'='*60}")

    # 資料集
    transform = get_clip_transform(img_size=224)
    dataset = get_dataset(dataset_name, args.data_root, split="test", transform=transform)
    sampler = FewShotEpisodeSampler(
        dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query
    )
    logger.info(f"Test dataset: {dataset}")

    # 模型
    clip_wrapper = CLIPWrapper(backbone=args.backbone, device=device)
    clip_lora = CLIPLoRA(
        clip_wrapper=clip_wrapper,
        r=args.lora_r,
        alpha=args.lora_alpha,
    )
    clip_lora.eval()

    # 載入 checkpoint
    ckpt_path = Path(args.checkpoint_dir) / f"{dataset_name}_{args.n_shot}shot_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "lora_state_dict" in ckpt:
            clip_lora.load_state_dict(ckpt["lora_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint: {ckpt_path}")
        else:
            clip_lora.load_state_dict(ckpt, strict=False)
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path}, using zero-shot CLIP")

    # ========== 1. 原始影像評估 ==========
    logger.info("\n[1/{}] Evaluating ORIGINAL images...".format(len(args.degrade_sizes) + 1))
    # 固定 seed 確保所有配置用相同的 episodes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    original_result = evaluate_with_degradation(
        clip_lora, sampler, args.n_episodes, device, degrade_size=None
    )
    original_result["degrade_size"] = "original (224)"
    original_result["dataset"] = dataset_name

    results = [original_result]

    # ========== 2. 各降解等級評估 ==========
    for i, deg_size in enumerate(sorted(args.degrade_sizes)):
        logger.info(
            f"\n[{i+2}/{len(args.degrade_sizes)+1}] "
            f"Evaluating DEGRADED images (degrade to {deg_size}×{deg_size})..."
        )
        # 重設 seed 確保相同的 episodes
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        deg_result = evaluate_with_degradation(
            clip_lora, sampler, args.n_episodes, device, degrade_size=deg_size
        )
        deg_result["degrade_size"] = f"{deg_size}×{deg_size}"
        deg_result["dataset"] = dataset_name

        # 計算相對於原始的變化
        deg_result["delta"] = deg_result["mean_acc"] - original_result["mean_acc"]

        results.append(deg_result)

    return results


def print_results_table(all_results: List[Dict], n_shot: int):
    """列印彙整的結果表格"""
    logger.info("\n" + "=" * 80)
    logger.info(f"DEGRADATION STUDY RESULTS ({n_shot}-shot)")
    logger.info("=" * 80)

    # 按資料集分組
    datasets = []
    for r in all_results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    # 表頭
    header = f"{'Dataset':<15} {'Resolution':<15} {'Accuracy':<20} {'PSNR (dB)':<12} {'Δ Acc':>8}"
    logger.info(header)
    logger.info("-" * 75)

    for ds in datasets:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        for r in ds_results:
            psnr_str = f"{r['mean_psnr']:.2f}" if "mean_psnr" in r else "N/A"
            delta_str = f"{r.get('delta', 0.0):+.2f}" if "delta" in r else "-"
            logger.info(
                f"{r['dataset']:<15} "
                f"{r['degrade_size']:<15} "
                f"{r['mean_acc']:.2f} ± {r['ci']:.2f}%     "
                f"{psnr_str:<12} "
                f"{delta_str:>8}"
            )
        logger.info("-" * 75)

    # 平均
    degrade_configs = []
    for r in all_results:
        if r["degrade_size"] not in degrade_configs:
            degrade_configs.append(r["degrade_size"])

    logger.info("\nAVERAGE ACROSS DATASETS:")
    for config in degrade_configs:
        config_results = [r for r in all_results if r["degrade_size"] == config]
        if config_results:
            avg_acc = np.mean([r["mean_acc"] for r in config_results])
            avg_delta = np.mean([r.get("delta", 0.0) for r in config_results])
            logger.info(
                f"  {config:<15} Avg Acc: {avg_acc:.2f}%  Δ: {avg_delta:+.2f}"
            )

    logger.info("\n結論分析提示：")
    logger.info("  - 若 56×56 降解導致 >5% 準確率下降 → 解析度確實是重要因素")
    logger.info("  - 若 112×112 降解影響很小 → 模型對輕度降解有一定魯棒性")
    logger.info("  - 若所有降解影響都很小 → 可能需要考慮 feature-level SR（方向 B）")


def save_results_csv(all_results: List[Dict], output_path: str):
    """將結果儲存為 CSV"""
    import csv
    fieldnames = ["dataset", "degrade_size", "mean_acc", "ci", "mean_psnr", "delta"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    logger.info(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    # 固定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 選擇資料集
    if args.dataset == "all":
        datasets = ["isic", "chestx", "eurosat", "crop_disease"]
    else:
        datasets = [args.dataset]

    # 執行實驗
    all_results = []
    for dataset_name in datasets:
        results = run_degradation_study(dataset_name, args, device)
        all_results.extend(results)

    # 輸出結果
    print_results_table(all_results, args.n_shot)

    # 可選 CSV 輸出
    if args.output_csv:
        save_results_csv(all_results, args.output_csv)
    else:
        # 預設存到 experiments/ 目錄
        default_csv = f"experiments/degradation_results_{args.n_shot}shot.csv"
        save_results_csv(all_results, default_csv)


if __name__ == "__main__":
    main()
