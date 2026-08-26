"""
evaluate.py
CC-CDFSL 評估主程式

對應論文評估設定：
  - 5-way 1-shot: 100 episodes
  - 5-way 5-shot: 400 episodes
  - 報告均值和 95% 信賴區間

用法：
    # 評估單一資料集
    python evaluate.py --dataset eurosat --checkpoint ./checkpoints/eurosat_5shot_best.pth --n_shot 5

    # 評估所有資料集
    python evaluate.py --dataset all --checkpoint_dir ./checkpoints --n_shot 5
"""
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform
from utils.metrics import compute_few_shot_accuracy, compute_confidence_interval
from losses.cycle_consistency import CyclicConsistencyLoss

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def augment_tensor_batch(images: torch.Tensor) -> torch.Tensor:
    """對 tensor 批次進行隨機水平翻轉和旋轉增強"""
    import torchvision.transforms.functional as TF
    aug_list = []
    for img in images:
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        aug_list.append(img)
    return torch.stack(aug_list)


def parse_args():
    parser = argparse.ArgumentParser(description="CC-CDFSL Evaluation")

    parser.add_argument("--dataset", type=str, default="eurosat",
                        choices=["eurosat", "isic", "chestx", "crop_disease", "all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="單一 checkpoint 路徑")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="評估所有資料集時的 checkpoint 目錄")

    # Few-shot 設定
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5,
                        choices=[1, 5])
    parser.add_argument("--n_query", type=int, default=15)

    # 評估集數
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="若為 None，則 1-shot 用 100，5-shot 用 400")

    # 模型設定
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    # Episode-specific Adaptation 設定
    parser.add_argument("--adapt", action="store_true",
                        help="是否對每個測試 Episode 進行支援集微調")
    parser.add_argument("--adapt_steps", type=int, default=100,
                        help="每個 Episode 微調的步數")
    parser.add_argument("--adapt_lr", type=float, default=1e-4,
                        help="微調的學習率")
    parser.add_argument("--lambda1", type=float, default=1.0,
                        help="T-I-T cycle loss 權重")
    parser.add_argument("--lambda2", type=float, default=0.5,
                        help="I-T-I cycle loss 權重")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Semantic Anchor top-k")
    parser.add_argument("--n_aug", type=int, default=4,
                        help="Augmentation 次數")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_dataset(
    dataset_name: str,
    checkpoint_path: str,
    args,
    device: str,
) -> dict:
    """對單一資料集進行完整評估"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: {dataset_name} ({args.n_shot}-shot)")
    if args.adapt:
        logger.info(f"Episode-specific Adaptation enabled ({args.adapt_steps} steps, lr={args.adapt_lr})")

    # 資料集
    transform = get_clip_transform(img_size=224)
    dataset = get_dataset(dataset_name, args.data_root, split="test", transform=transform)
    logger.info(f"Test dataset: {dataset}")

    # 取樣器
    sampler = FewShotEpisodeSampler(
        dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query
    )

    # 模型
    clip_wrapper = CLIPWrapper(backbone=args.backbone, device=device)
    clip_lora = CLIPLoRA(
        clip_wrapper=clip_wrapper,
        r=args.lora_r,
        alpha=args.lora_alpha,
    )
    clip_lora.eval()

    # 載入 checkpoint
    if checkpoint_path and Path(checkpoint_path).exists() and checkpoint_path.lower() != "none":
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "lora_state_dict" in ckpt:
            missing, unexpected = clip_lora.load_state_dict(ckpt["lora_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(f"  epoch={ckpt.get('epoch', '?')}, best_acc={ckpt.get('best_acc', '?'):.2f}%")
        else:
            missing, unexpected = clip_lora.load_state_dict(ckpt, strict=False)
    else:
        logger.warning(f"No checkpoint loaded, using zero-shot CLIP as base")

    # 評估 episodes
    n_episodes = args.n_episodes
    if n_episodes is None:
        n_episodes = 100 if args.n_shot == 1 else 400

    logger.info(f"Running {n_episodes} episodes...")
    accs = []

    # 初始化 cycle consistency loss (若要適應)
    cc_loss_fn = CyclicConsistencyLoss(top_k=args.top_k) if args.adapt else None

    for ep in range(n_episodes):
        episode = sampler.sample()

        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        class_names  = episode["class_names"]

        if args.adapt:
            # 1. 備份 LoRA 參數
            lora_state_backup = {k: v.cpu().clone() for k, v in clip_lora.state_dict().items() if "lora_" in k}

            # 2. 建立此 episode 專屬的優化器
            trainable_params = clip_lora.trainable_parameters()
            optimizer = torch.optim.Adam(trainable_params, lr=args.adapt_lr, weight_decay=5e-4)

            # 3. 提取 support set
            support_images = episode["support_images"].to(device)
            support_labels = episode["support_labels"].to(device)

            # 4. 微調循環 (Adaptation Loop)
            clip_lora.train()
            for step in range(args.adapt_steps):
                optimizer.zero_grad()

                # 文字特徵不需要 LoRA 梯度
                with torch.no_grad():
                    text_feat = clip_lora.encode_text(class_names)  # [C, d]

                # Support 特徵提取 (需要梯度)
                sup_cls, sup_patches = clip_lora.encode_image_with_patches(support_images)

                # CE Loss
                logit_scale = clip_lora.clip.model.logit_scale.exp()
                sup_logits = sup_cls @ text_feat.T * logit_scale
                loss_ce = F.cross_entropy(sup_logits, support_labels)

                # CC-CDFSL Loss
                aug_list = []
                for _ in range(args.n_aug):
                    aug_imgs = augment_tensor_batch(support_images)
                    aug_list.append(aug_imgs)
                aug_images = torch.cat(aug_list, dim=0)

                _, aug_patches = clip_lora.encode_image_with_patches(aug_images)

                loss_tit, loss_iti = cc_loss_fn(
                    text_feat=text_feat,
                    patch_feat_orig=sup_patches,
                    patch_feat_aug=aug_patches,
                )

                loss = loss_ce + args.lambda1 * loss_tit + args.lambda2 * loss_iti
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            # 5. 評估 (Evaluation)
            clip_lora.eval()
            with torch.no_grad():
                text_feat = clip_lora.encode_text(class_names)
                cls_feat, _ = clip_lora.encode_image_with_patches(query_images)
                logits = cls_feat @ text_feat.T
                acc = compute_few_shot_accuracy(logits, query_labels)
                accs.append(acc)

            # 6. 還原權重
            clip_lora.load_state_dict(lora_state_backup, strict=False)

        else:
            # 正常的路徑 (No adaptation)
            with torch.no_grad():
                text_feat = clip_lora.encode_text(class_names)
                cls_feat, _ = clip_lora.encode_image_with_patches(query_images)
                logits = cls_feat @ text_feat.T
                acc = compute_few_shot_accuracy(logits, query_labels)
                accs.append(acc)

        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            mean_tmp, ci_tmp = compute_confidence_interval(accs)
            logger.info(f"  Episode {ep+1}/{n_episodes}: {mean_tmp:.2f} ± {ci_tmp:.2f}%")

    mean_acc, ci = compute_confidence_interval(accs)
    logger.info(f"\nResult: {mean_acc:.2f} ± {ci:.2f}% ({n_episodes} episodes)")

    return {
        "dataset": dataset_name,
        "n_shot": args.n_shot,
        "mean_acc": mean_acc,
        "ci": ci,
        "n_episodes": n_episodes,
    }


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "all":
        datasets = ["isic", "chestx", "eurosat", "crop_disease"]
    else:
        datasets = [args.dataset]

    results = []
    for dataset_name in datasets:
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_path = str(
                Path(args.checkpoint_dir)
                / f"{dataset_name}_{args.n_shot}shot_best.pth"
            )

        result = evaluate_dataset(dataset_name, ckpt_path, args, device)
        results.append(result)

    # 彙總
    logger.info("\n" + "="*60)
    logger.info("Final Summary:")
    logger.info("="*60)
    accs = []
    for r in results:
        logger.info(
            f"  {r['dataset']:15s} {r['n_shot']}-shot: "
            f"{r['mean_acc']:.2f} ± {r['ci']:.2f}%"
        )
        accs.append(r["mean_acc"])

    if len(results) > 1:
        logger.info(f"  {'Average':15s}: {np.mean(accs):.2f}%")

    # 論文表格格式輸出
    logger.info("\n[Table Format]")
    logger.info("Dataset\t\tMean±CI")
    for r in results:
        logger.info(f"{r['dataset']}\t{r['mean_acc']:.2f}±{r['ci']:.2f}")


if __name__ == "__main__":
    main()
