"""
experiments/compare_fsr_vs_baseline.py
Feature SR vs Baseline 準確率比較實驗

功能：
  1. 對所有有 checkpoint 的資料集，同時評估：
     - Baseline (14×14 patches)
     - Feature SR (28×28 patches)
  2. 顯示準確率差異（Delta）
  3. 輸出 CSV 結果供後續分析

用法：
    # 評估所有有 checkpoint 的資料集（5-shot）
    python experiments/compare_fsr_vs_baseline.py --n_shot 5 --n_episodes 200

    # 快速測試（episoe 少）
    python experiments/compare_fsr_vs_baseline.py --n_shot 5 --n_episodes 50

    # 指定特定資料集
    python experiments/compare_fsr_vs_baseline.py --dataset eurosat --n_shot 5 --n_episodes 200
"""
import sys
import argparse
import logging
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from models.feature_sr import FeatureSRModule
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform
from utils.metrics import compute_few_shot_accuracy, compute_confidence_interval

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Feature SR vs Baseline Comparison")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["eurosat", "isic", "chestx", "crop_disease", "all"])
    parser.add_argument("--n_shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="評估的 episode 數（論文用 400，快速測試用 50-100）")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--output_csv", type=str, default=None,
                        help="輸出 CSV 路徑（預設自動命名）")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    dataset_name: str,
    n_shot: int,
    n_way: int,
    n_query: int,
    n_episodes: int,
    device: str,
    backbone: str = "ViT-B/16",
    lora_r: int = 4,
    lora_alpha: float = 1.0,
    data_root: str = str(DATA_ROOT),
    use_feature_sr: bool = False,
) -> dict:
    """
    載入 checkpoint 並評估準確率。

    Returns:
        dict with keys: mean_acc, ci, mode, dataset, n_shot
    """
    mode = "Feature-SR (28×28)" if use_feature_sr else "Baseline (14×14)"
    logger.info(f"  Loading checkpoint: {ckpt_path.name}")

    # 載入 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_epoch = ckpt.get("epoch", "?")
    saved_best = ckpt.get("best_acc", "?")
    logger.info(f"  Checkpoint: epoch={saved_epoch}, best_acc={saved_best:.2f}%"
                if isinstance(saved_best, float) else f"  Checkpoint: epoch={saved_epoch}")

    # ─── 建立模型 ───
    clip_wrapper = CLIPWrapper(backbone=backbone, device=device)
    clip_lora = CLIPLoRA(
        clip_wrapper=clip_wrapper,
        r=lora_r,
        alpha=lora_alpha,
        dropout=0.0,
    )

    # 載入 LoRA 權重
    lora_sd = ckpt.get("lora_state_dict", {})
    missing, unexpected = clip_lora.load_state_dict(lora_sd, strict=False)
    if missing:
        logger.debug(f"  Missing keys: {len(missing)}")

    # Feature SR 模組（若有）
    feature_sr = None
    if use_feature_sr and "feature_sr_state_dict" in ckpt:
        # 從 checkpoint args 取得 FSR 超參數
        saved_args = ckpt.get("args", {})
        sr_scale = saved_args.get("sr_scale", 2)
        sr_layers = saved_args.get("sr_refiner_layers", 2)
        sr_heads = saved_args.get("sr_refiner_heads", 8)

        feature_sr = FeatureSRModule(
            feat_dim=clip_wrapper.output_dim,
            input_size=14,
            scale=sr_scale,
            refiner_layers=sr_layers,
            refiner_heads=sr_heads,
        ).to(device)
        feature_sr.load_state_dict(ckpt["feature_sr_state_dict"])
        feature_sr.eval()
        logger.info(f"  Feature SR loaded: scale={sr_scale}, layers={sr_layers}, heads={sr_heads}")
    elif use_feature_sr:
        logger.warning("  ⚠️  use_feature_sr=True but no feature_sr_state_dict in checkpoint! "
                       "Running with random FSR weights.")
        feature_sr = FeatureSRModule(
            feat_dim=clip_wrapper.output_dim,
            input_size=14,
            scale=2,
        ).to(device)
        feature_sr.eval()

    clip_lora.eval()

    # ─── 建立資料集 ───
    transform = get_clip_transform(img_size=224)
    test_dataset = get_dataset(dataset_name, data_root, split="test", transform=transform)
    test_sampler = FewShotEpisodeSampler(
        test_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query
    )

    # ─── 評估迴圈 ───
    accs = []
    for ep_idx in range(n_episodes):
        episode = test_sampler.sample()
        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        class_names  = episode["class_names"]

        # 文字特徵
        text_feat = clip_lora.encode_text(class_names)   # [C, d]

        # 影像特徵
        q_cls, _ = clip_lora.encode_image_with_patches(query_images)
        q_logits = q_cls @ text_feat.T

        acc = compute_few_shot_accuracy(q_logits, query_labels)
        accs.append(acc)

        if (ep_idx + 1) % 50 == 0:
            running_mean = np.mean(accs)
            logger.info(f"    [{ep_idx+1}/{n_episodes}] Running acc: {running_mean:.2f}%")

    mean_acc, ci = compute_confidence_interval(accs)
    return {
        "dataset": dataset_name,
        "n_shot": n_shot,
        "mode": mode,
        "mean_acc": mean_acc,
        "ci": ci,
        "saved_epoch": saved_epoch,
        "saved_best_acc": saved_best if isinstance(saved_best, float) else 0.0,
    }


def get_checkpoint_pairs(checkpoint_dir: Path, n_shot: int) -> list:
    """
    掃描 checkpoint 目錄，找出有 baseline 或 fsr checkpoint 的資料集。

    Returns:
        list of dict: {dataset, baseline_ckpt, fsr_ckpt}
    """
    datasets = ["isic", "chestx", "eurosat", "crop_disease"]
    pairs = []

    for ds in datasets:
        baseline_ckpt = checkpoint_dir / f"{ds}_{n_shot}shot_best.pth"
        fsr_ckpt = checkpoint_dir / f"{ds}_{n_shot}shot_fsr_best.pth"

        has_baseline = baseline_ckpt.exists()
        has_fsr = fsr_ckpt.exists()

        if has_baseline or has_fsr:
            pairs.append({
                "dataset": ds,
                "baseline_ckpt": baseline_ckpt if has_baseline else None,
                "fsr_ckpt": fsr_ckpt if has_fsr else None,
            })

    return pairs


def print_comparison_table(results: list):
    """漂亮地印出比較表格"""
    print("\n" + "="*75)
    print(f"{'Feature SR vs Baseline 準確率比較':^75}")
    print("="*75)
    print(f"{'資料集':<15} {'Baseline (14×14)':<22} {'Feature SR (28×28)':<22} {'Delta':<10}")
    print("-"*75)

    for r in results:
        ds = r["dataset"]
        bl = r.get("baseline")
        fsr = r.get("fsr")

        if bl and fsr:
            bl_str = f"{bl['mean_acc']:.2f} ± {bl['ci']:.2f}%"
            fsr_str = f"{fsr['mean_acc']:.2f} ± {fsr['ci']:.2f}%"
            delta = fsr['mean_acc'] - bl['mean_acc']
            delta_str = f"{delta:+.2f}% {'✅' if delta > 0 else '❌'}"
        elif bl:
            bl_str = f"{bl['mean_acc']:.2f} ± {bl['ci']:.2f}%"
            fsr_str = "N/A (no ckpt)"
            delta_str = "—"
        elif fsr:
            bl_str = "N/A (no ckpt)"
            fsr_str = f"{fsr['mean_acc']:.2f} ± {fsr['ci']:.2f}%"
            delta_str = "—"
        else:
            continue

        print(f"{ds:<15} {bl_str:<22} {fsr_str:<22} {delta_str}")

    print("="*75)

    # 平均（僅計算兩者都有的）
    deltas = []
    for r in results:
        if r.get("baseline") and r.get("fsr"):
            deltas.append(r["fsr"]["mean_acc"] - r["baseline"]["mean_acc"])
    if deltas:
        print(f"{'Average Delta':>60} {np.mean(deltas):+.2f}%")
    print("="*75)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    checkpoint_dir = Path(args.checkpoint_dir)
    n_shot = args.n_shot

    # ─── 確定要評估哪些資料集 ───
    if args.dataset == "all":
        pairs = get_checkpoint_pairs(checkpoint_dir, n_shot)
    else:
        ds = args.dataset
        baseline_ckpt = checkpoint_dir / f"{ds}_{n_shot}shot_best.pth"
        fsr_ckpt = checkpoint_dir / f"{ds}_{n_shot}shot_fsr_best.pth"
        pairs = [{
            "dataset": ds,
            "baseline_ckpt": baseline_ckpt if baseline_ckpt.exists() else None,
            "fsr_ckpt": fsr_ckpt if fsr_ckpt.exists() else None,
        }]

    logger.info(f"Found {len(pairs)} datasets to evaluate ({n_shot}-shot, {args.n_episodes} episodes)")
    for p in pairs:
        has_bl = "✅" if p["baseline_ckpt"] else "❌"
        has_fsr = "✅" if p["fsr_ckpt"] else "❌"
        logger.info(f"  {p['dataset']:15s} Baseline:{has_bl}  FSR:{has_fsr}")

    # ─── 逐資料集評估 ───
    comparison_results = []

    for pair in pairs:
        ds = pair["dataset"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"Evaluating: {ds} ({n_shot}-shot)")
        logger.info(f"{'─'*60}")

        row = {"dataset": ds, "baseline": None, "fsr": None}

        # Baseline
        if pair["baseline_ckpt"]:
            logger.info(f"[Baseline]")
            result_bl = evaluate_checkpoint(
                ckpt_path=pair["baseline_ckpt"],
                dataset_name=ds,
                n_shot=n_shot,
                n_way=args.n_way,
                n_query=args.n_query,
                n_episodes=args.n_episodes,
                device=device,
                backbone=args.backbone,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                data_root=args.data_root,
                use_feature_sr=False,
            )
            row["baseline"] = result_bl
            logger.info(f"  → Baseline: {result_bl['mean_acc']:.2f} ± {result_bl['ci']:.2f}%")

        # Feature SR
        if pair["fsr_ckpt"]:
            logger.info(f"[Feature SR]")
            result_fsr = evaluate_checkpoint(
                ckpt_path=pair["fsr_ckpt"],
                dataset_name=ds,
                n_shot=n_shot,
                n_way=args.n_way,
                n_query=args.n_query,
                n_episodes=args.n_episodes,
                device=device,
                backbone=args.backbone,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                data_root=args.data_root,
                use_feature_sr=True,
            )
            row["fsr"] = result_fsr
            logger.info(f"  → Feature SR: {result_fsr['mean_acc']:.2f} ± {result_fsr['ci']:.2f}%")

        comparison_results.append(row)

    # ─── 印出比較表格 ───
    print_comparison_table(comparison_results)

    # ─── 儲存 CSV ───
    out_csv = args.output_csv
    if out_csv is None:
        exp_dir = Path(__file__).parent
        out_csv = str(exp_dir / f"fsr_comparison_{n_shot}shot.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mode", "n_shot", "mean_acc", "ci", "delta"])
        for row in comparison_results:
            bl = row.get("baseline")
            fsr = row.get("fsr")
            if bl:
                writer.writerow([row["dataset"], "baseline", n_shot,
                                  f"{bl['mean_acc']:.4f}", f"{bl['ci']:.4f}", ""])
            if fsr:
                delta = (fsr["mean_acc"] - bl["mean_acc"]) if bl else ""
                writer.writerow([row["dataset"], "feature_sr", n_shot,
                                  f"{fsr['mean_acc']:.4f}", f"{fsr['ci']:.4f}",
                                  f"{delta:.4f}" if isinstance(delta, float) else ""])

    logger.info(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
