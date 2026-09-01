"""
train.py
CC-CDFSL 訓練主程式（支援 Feature-level SR 擴展）

對應論文 Section 4.1 的實驗設定：
  - ViT-B/16 backbone
  - 100 epochs 訓練
  - 5-way 1-shot / 5-shot 設定
  - 在 CropDiseases, EuroSAT, ISIC2018, ChestX 上評估
  - [NEW] Feature-level SR (14×14 → 28×28 patch features)

用法：
    # 執行 Baseline CC-CDFSL
    python train.py --dataset eurosat --n_shot 5 --epochs 100

    # 執行 Feature-SR CC-CDFSL (我們的改進方法 ⭐)
    python train.py --dataset eurosat --n_shot 5 --epochs 100 --use_feature_sr
"""
import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# 把上層目錄加入 path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from models.feature_sr import FeatureSRModule, CrossResolutionConsistencyLoss
from losses.cycle_consistency import CyclicConsistencyLoss
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform, get_aug_transform
from utils.metrics import compute_accuracy, compute_confidence_interval, compute_few_shot_accuracy

# ─────────────────────────────────────────────
# 日誌設定
# ─────────────────────────────────────────────
logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 引數解析
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CC-CDFSL Training with Feature-SR")

    # 資料設定
    parser.add_argument("--dataset", type=str, default="eurosat",
                        choices=["eurosat", "isic", "chestx", "crop_disease", "all"],
                        help="目標資料集")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="資料根目錄")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路徑")

    # Few-shot 設定
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5,
                        help="每類 support 樣本數 (1 or 5)")
    parser.add_argument("--n_query", type=int, default=15)

    # 模型設定
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # CC-CDFSL 設定
    parser.add_argument("--lambda1", type=float, default=1.0,
                        help="T-I-T cycle loss 權重")
    parser.add_argument("--lambda2", type=float, default=0.5,
                        help="I-T-I cycle loss 權重")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Semantic Anchor top-k")
    parser.add_argument("--n_aug", type=int, default=4,
                        help="Augmentation 次數")

    # Feature-level SR 擴展設定 ⭐
    parser.add_argument("--use_feature_sr", action="store_true",
                        help="啟用 Feature-level SR 模組 (14x14 -> 28x28 patch features)")
    parser.add_argument("--sr_scale", type=int, default=2,
                        help="Feature SR 上採樣倍率 (預設 2: 14x14 -> 28x28)")
    parser.add_argument("--sr_refiner_layers", type=int, default=2,
                        help="Feature Refiner Transformer block 數")
    parser.add_argument("--sr_refiner_heads", type=int, default=8,
                        help="Feature Refiner attention heads")
    parser.add_argument("--lambda3", type=float, default=0.3,
                        help="Cross-Resolution Consistency loss 權重")

    # 訓練設定
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes_per_epoch", type=int, default=100,
                        help="每個 epoch 的 episode 數")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="線性 warmup 的 epoch 數（預設 0 = 無 warmup）")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"])

    # 評估設定
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="每幾個 epoch 評估一次")
    parser.add_argument("--n_eval_episodes", type=int, default=100,
                        help="評估時的 episode 數")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--strict_few_shot", action="store_true",
                        help="僅使用固定 n_shot 數量的影像進行訓練（符合論文少樣本目標域微調設定）")
    parser.add_argument("--bscd_mode", action="store_true",
                        help="啟用 BSCD-FSL 官方協定：全量資料進 episodic pool，無 train/test 80/20 split")
    parser.add_argument("--use_prompt_ensemble", action="store_true",
                        help="啟用領域專屬 Multi-Prompt 特徵集成")
    parser.add_argument("--use_tta", action="store_true",
                        help="評估時啟用測試期多視角特徵增強 (Test-Time Augmentation)")

    return parser.parse_args()


# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_optimizer(params, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


# ─────────────────────────────────────────────
# 一個 Episode 的訓練步驟
# ─────────────────────────────────────────────
def train_episode(
    clip_lora: CLIPLoRA,
    cc_loss_fn: CyclicConsistencyLoss,
    optimizer: torch.optim.Optimizer,
    episode: dict,
    aug_transform,
    device: str,
    lambda1: float,
    lambda2: float,
    n_aug: int,
    feature_sr: Optional[FeatureSRModule] = None,
    cr_loss_fn: Optional[CrossResolutionConsistencyLoss] = None,
    lambda3: float = 0.3,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    dataset_name: Optional[str] = None,
    use_prompt_ensemble: bool = False,
) -> dict:
    """
    一個 Few-shot Episode 的訓練步驟。
    """
    clip_lora.train()
    if feature_sr is not None:
        feature_sr.train()

    optimizer.zero_grad()

    # 移動到設備
    support_images = episode["support_images"].to(device)   # [K*N, 3, H, W]
    support_labels = episode["support_labels"].to(device)   # [K*N]
    query_images   = episode["query_images"].to(device)     # [K*Q, 3, H, W]
    query_labels   = episode["query_labels"].to(device)     # [K*Q]
    class_names    = episode["class_names"]                 # [K]

    # ===== 1. 文字特徵（不需要梯度） =====
    with torch.no_grad():
        text_feat = clip_lora.encode_text(
            class_names,
            dataset_name=dataset_name,
            use_ensemble=use_prompt_ensemble,
        )      # [C, d]

    # 使用 PyTorch 自動混合精度 (AMP) 進行加速
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        # ===== 2. Support Set 特徵提取（需要梯度）=====
        sup_cls, sup_patches = clip_lora.encode_image_with_patches(support_images)
        # sup_cls:     [K*N, d]
        # sup_patches: [K*N, P, d]  (P=196)

        # ===== 2b. Feature SR 上採樣 (14x14 -> 28x28) =====
        if feature_sr is not None:
            sup_patches_sr, sup_patches_orig = feature_sr(sup_patches, return_both=True)
            # sup_patches_sr: [K*N, 784, d]
        else:
            sup_patches_sr = sup_patches
            sup_patches_orig = None

        # ===== 3. Cross-entropy Loss =====
        logit_scale = clip_lora.clip.model.logit_scale.exp()
        sup_logits = sup_cls @ text_feat.T * logit_scale      # [K*N, C]
        loss_ce = F.cross_entropy(sup_logits, support_labels)

        # ===== 4. Augmentation for I-T-I =====
        aug_list = []
        for _ in range(n_aug):
            aug_imgs = augment_tensor_batch(support_images)
            aug_list.append(aug_imgs)
        aug_images = torch.cat(aug_list, dim=0)                # [K*N*n_aug, 3, H, W]

        with torch.no_grad():
            _, aug_patches = clip_lora.encode_image_with_patches(aug_images)
        # aug_patches: [K*N*n_aug, P, d]

        if feature_sr is not None:
            with torch.no_grad():
                aug_patches_sr, _ = feature_sr(aug_patches, return_both=False)
        else:
            aug_patches_sr = aug_patches

        # ===== 5. Cycle Consistency Loss =====
        loss_tit, loss_iti = cc_loss_fn(
            text_feat=text_feat,
            patch_feat_orig=sup_patches_sr,    # [K*N, P_sr, d]
            patch_feat_aug=aug_patches_sr,     # [K*N*n_aug, P_sr, d]
        )

        # ===== 5b. Cross-Resolution Consistency Loss =====
        loss_cr = torch.tensor(0.0, device=device)
        if feature_sr is not None and cr_loss_fn is not None and sup_patches_orig is not None:
            loss_cr = cr_loss_fn(sup_patches_sr, sup_patches_orig)

        # ===== 6. 總損失 =====
        loss = loss_ce + lambda1 * loss_tit + lambda2 * loss_iti + lambda3 * loss_cr

    # 反向傳播與參數優化
    trainable_params = list(clip_lora.trainable_parameters())
    if feature_sr is not None:
        trainable_params.extend(list(feature_sr.parameters()))

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

    # ===== 7. Query Set 準確率 =====
    with torch.no_grad():
        q_cls, _ = clip_lora.encode_image_with_patches(query_images)
        q_logits = q_cls @ text_feat.T
        acc = compute_few_shot_accuracy(q_logits, query_labels)

    return {
        "loss": loss.item(),
        "loss_ce": loss_ce.item(),
        "loss_tit": loss_tit.item(),
        "loss_iti": loss_iti.item(),
        "loss_cr": loss_cr.item() if isinstance(loss_cr, torch.Tensor) else loss_cr,
        "accuracy": acc,
    }


def augment_tensor_batch(images: torch.Tensor) -> torch.Tensor:
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


# ─────────────────────────────────────────────
# 評估函數
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    clip_lora: CLIPLoRA,
    sampler: FewShotEpisodeSampler,
    n_episodes: int,
    device: str,
    dataset_name: Optional[str] = None,
    use_prompt_ensemble: bool = False,
    use_tta: bool = False,
) -> dict:
    clip_lora.eval()
    accs = []

    for _ in range(n_episodes):
        episode = sampler.sample()

        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        class_names  = episode["class_names"]

        text_feat = clip_lora.encode_text(
            class_names,
            dataset_name=dataset_name,
            use_ensemble=use_prompt_ensemble,
        )

        if use_tta:
            q_cls_std, _ = clip_lora.encode_image_with_patches(query_images)
            q_cls_flip, _ = clip_lora.encode_image_with_patches(torch.flip(query_images, dims=[-1]))
            q_cls = F.normalize((F.normalize(q_cls_std, dim=-1) + F.normalize(q_cls_flip, dim=-1)) / 2.0, dim=-1)
        else:
            q_cls, _ = clip_lora.encode_image_with_patches(query_images)
            q_cls = F.normalize(q_cls, dim=-1)

        q_logits = q_cls @ text_feat.T

        acc = compute_few_shot_accuracy(q_logits, query_labels)
        accs.append(acc)

    mean_acc, ci = compute_confidence_interval(accs)
    return {"mean_acc": mean_acc, "ci": ci}


# ─────────────────────────────────────────────
# 主訓練迴圈
# ─────────────────────────────────────────────
def train_on_dataset(
    dataset_name: str,
    args,
    device: str,
) -> dict:
    """對單一資料集進行訓練"""
    mode_str = "Feature-SR (28x28)" if args.use_feature_sr else "Baseline (14x14)"
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name}, n_shot={args.n_shot}, Mode: {mode_str}")
    logger.info(f"{'='*60}")

    # ========== 建立資料集 ==========
    transform = get_clip_transform(img_size=224)
    train_n_shot = args.n_shot if getattr(args, "strict_few_shot", False) else None
    bscd_mode = getattr(args, "bscd_mode", False)

    train_dataset = get_dataset(dataset_name, args.data_root, split="train", transform=transform,
                                n_shot=train_n_shot, bscd_mode=bscd_mode)

    if bscd_mode:
        # BSCD-FSL 官方協定：訓練和評估用同一個全量 pool
        test_dataset = train_dataset
        logger.info(f"[BSCD模式] Train = Test = 全量資料 ({len(train_dataset)} 張)")
    else:
        test_dataset = get_dataset(dataset_name, args.data_root, split="test", transform=transform,
                                   bscd_mode=False)

    logger.info(f"Train: {train_dataset}")
    logger.info(f"Test:  {test_dataset}")

    # ========== 建立取樣器 ==========
    if getattr(args, "strict_few_shot", False):
        train_sampler = FewShotEpisodeSampler(
            train_dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, query_dataset=test_dataset
        )
    else:
        train_sampler = FewShotEpisodeSampler(
            train_dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query
        )
    test_sampler = FewShotEpisodeSampler(
        test_dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query
    )

    # ========== 建立模型 ==========
    logger.info(f"Loading CLIP {args.backbone}...")
    clip_wrapper = CLIPWrapper(backbone=args.backbone, device=device)

    logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    clip_lora = CLIPLoRA(
        clip_wrapper=clip_wrapper,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Feature SR 模組
    feature_sr = None
    cr_loss_fn = None
    if args.use_feature_sr:
        logger.info(f"Initializing Feature SR Module (scale={args.sr_scale}, 14x14 -> 28x28)...")
        feature_sr = FeatureSRModule(
            feat_dim=clip_wrapper.output_dim,
            input_size=14,
            scale=args.sr_scale,
            refiner_layers=args.sr_refiner_layers,
            refiner_heads=args.sr_refiner_heads,
        ).to(device)
        cr_loss_fn = CrossResolutionConsistencyLoss(input_size=14, scale=args.sr_scale).to(device)

    trainable_params = list(clip_lora.trainable_parameters())
    if feature_sr is not None:
        trainable_params.extend(list(feature_sr.parameters()))

    total_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ========== 損失和優化器 ==========
    cc_loss_fn = CyclicConsistencyLoss(top_k=args.top_k)
    optimizer = build_optimizer(trainable_params, args)

    # Cosine Annealing with optional linear warmup
    warmup_epochs = getattr(args, "warmup_epochs", 0)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
    )

    if warmup_epochs > 0:
        # Linear warmup: lr ramps from 0 to base_lr over warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        logger.info(f"LR Schedule: Linear warmup ({warmup_epochs} epochs) + Cosine decay")
    else:
        scheduler = cosine_scheduler
        logger.info(f"LR Schedule: Cosine decay (no warmup)")

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    if scaler is not None:
        logger.info("FP16 Automatic Mixed Precision (AMP) enabled.")

    # ========== 初始評估 ==========
    logger.info("Initial evaluation (before training)...")
    init_result = evaluate(clip_lora, test_sampler, n_episodes=50, device=device)
    logger.info(f"  Baseline: {init_result['mean_acc']:.2f} ± {init_result['ci']:.2f}%")

    # ========== 訓練迴圈 ==========
    best_acc = 0.0
    best_epoch = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fsr" if args.use_feature_sr else ""

    for epoch in range(1, args.epochs + 1):
        epoch_stats = {"loss": 0, "loss_ce": 0, "loss_tit": 0, "loss_iti": 0, "loss_cr": 0, "accuracy": 0}

        for ep_idx in range(args.episodes_per_epoch):
            episode = train_sampler.sample()
            stats = train_episode(
                clip_lora=clip_lora,
                cc_loss_fn=cc_loss_fn,
                optimizer=optimizer,
                episode=episode,
                aug_transform=None,
                device=device,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                n_aug=args.n_aug,
                feature_sr=feature_sr,
                cr_loss_fn=cr_loss_fn,
                lambda3=args.lambda3,
                scaler=scaler,
                dataset_name=dataset_name,
                use_prompt_ensemble=getattr(args, "use_prompt_ensemble", False),
            )
            for k in epoch_stats:
                epoch_stats[k] += stats[k]

        for k in epoch_stats:
            epoch_stats[k] /= args.episodes_per_epoch

        scheduler.step()

        if epoch % args.log_interval == 0 or epoch == 1:
            cr_str = f", CR:{epoch_stats['loss_cr']:.4f}" if args.use_feature_sr else ""
            logger.info(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Loss: {epoch_stats['loss']:.4f} "
                f"(CE:{epoch_stats['loss_ce']:.4f}, "
                f"TIT:{epoch_stats['loss_tit']:.4f}, "
                f"ITI:{epoch_stats['loss_iti']:.4f}{cr_str}) "
                f"Acc: {epoch_stats['accuracy']:.2f}%"
            )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_result = evaluate(
                clip_lora, test_sampler,
                n_episodes=args.n_eval_episodes,
                device=device,
                dataset_name=dataset_name,
                use_prompt_ensemble=getattr(args, "use_prompt_ensemble", False),
                use_tta=getattr(args, "use_tta", False),
            )
            mean_acc = eval_result["mean_acc"]
            ci = eval_result["ci"]
            logger.info(
                f"  [Eval] Epoch {epoch}: {mean_acc:.2f} ± {ci:.2f}%"
                + (" ✓ NEW BEST!" if mean_acc > best_acc else "")
            )

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = epoch
                ckpt_path = save_dir / f"{dataset_name}_{args.n_shot}shot{suffix}_best.pth"
                save_dict = {
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "lora_state_dict": {
                        k: v for k, v in clip_lora.state_dict().items()
                        if "lora_" in k
                    },
                    "args": vars(args),
                }
                if feature_sr is not None:
                    save_dict["feature_sr_state_dict"] = feature_sr.state_dict()

                torch.save(save_dict, ckpt_path)
                logger.info(f"  Saved best checkpoint: {ckpt_path}")

    logger.info(f"\nBest Acc: {best_acc:.2f}% at Epoch {best_epoch}")

    # ========== 最終評估 ==========
    n_final = 400 if args.n_shot == 5 else 100
    final_ckpt = save_dir / f"{dataset_name}_{args.n_shot}shot{suffix}_best.pth"
    if final_ckpt.exists():
        ckpt = torch.load(final_ckpt, map_location=device)
        clip_lora.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if feature_sr is not None and "feature_sr_state_dict" in ckpt:
            feature_sr.load_state_dict(ckpt["feature_sr_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    final_result = evaluate(
        clip_lora, test_sampler,
        n_episodes=n_final,
        device=device,
        dataset_name=dataset_name,
        use_prompt_ensemble=getattr(args, "use_prompt_ensemble", False),
        use_tta=getattr(args, "use_tta", False),
    )
    logger.info(
        f"Final ({mode_str}): {final_result['mean_acc']:.2f} ± {final_result['ci']:.2f}%"
        f" (n_episodes={n_final})"
    )

    return {
        "dataset": dataset_name,
        "n_shot": args.n_shot,
        "mode": mode_str,
        "mean_acc": final_result["mean_acc"],
        "ci": final_result["ci"],
    }


def main():
    args = parse_args()

    config = load_config(args.config)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    logger.info(f"Device: {device}")

    set_seed(args.seed)

    datasets = ["isic", "chestx", "eurosat", "crop_disease"] if args.dataset == "all" else [args.dataset]

    results = []
    for dataset_name in datasets:
        result = train_on_dataset(dataset_name, args, device)
        results.append(result)

    if len(results) > 1:
        logger.info("\n" + "="*60)
        logger.info("Summary:")
        logger.info("="*60)
        accs = []
        for r in results:
            logger.info(
                f"  {r['dataset']:15s} {r['n_shot']}-shot ({r['mode']}): "
                f"{r['mean_acc']:.2f} ± {r['ci']:.2f}%"
            )
            accs.append(r["mean_acc"])
        logger.info(f"  {'Average':15s}: {np.mean(accs):.2f}%")


if __name__ == "__main__":
    main()
