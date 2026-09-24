"""
train_episodic.py
Episode-level fine-tuning protocol，對齊 CC-CDFSL (CVPR 2026) / StepSPT (TPAMI 2025)
的 source-free CDFSL 設定：

  每個測試 episode：
    1. LoRA 重置回預訓練 CLIP（B=0 ⇒ 模型完全等於原始 CLIP）
    2. 只用該 episode 的 support set（N·K 張）微調 `--steps` 步
    3. 用微調後的模型分類該 episode 的 query set
  重複 100（1-shot）/ 400（5-shot）個 episode，回報平均 ± 95% CI。

LoRA / 優化器預設值照 CLIP-LoRA 官方實作 (MaxZanella/CLIP-LoRA, run_utils.py / lora.py)：
  視覺+文字兩個 encoder、q/k/v 各一組 LoRA、所有層、r=2、alpha=1、dropout=0.25、
  AdamW(lr=2e-4, wd=1e-2, betas=(0.9,0.999))、CosineAnnealingLR(eta_min=1e-6)、batch 32、
  CE 訓練增強 = RandomResizedCrop(scale=0.08–1) + 水平翻轉、prompt "a photo of a {}"。

λ1=λ2=0 且不開 --use_feature_sr 時即為純 CLIP-LoRA baseline（用來先驗證協定對齊）。

Cycle-consistency 的 patch 語料：可微分那次前向用的是 CE 增強後的 support view，
Semantic Anchor 的 A 個增強視角（水平翻轉 / ±15° 旋轉，與 train.py 相同）也從這個 view 產生、
在 no_grad 下取 patch，與 train.py 的做法一致。

每個 episode 用 (seed, episode index) 各自設定亂數種子，因此結果可逐 episode 重現；
每個 episode 的準確率即時寫入 .jsonl，中斷後重跑同一指令會自動從斷點續跑。
"""
import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from models.feature_sr import FeatureSRModule, CrossResolutionConsistencyLoss
from losses.cycle_consistency import CyclicConsistencyLoss
from utils.augmentation import get_clip_transform
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.metrics import compute_confidence_interval, compute_few_shot_accuracy
from train import augment_tensor_batch

logging.basicConfig(format="[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Episode-level fine-tuning (CC-CDFSL / StepSPT protocol)")
    p.add_argument("--dataset", required=True, choices=["eurosat", "isic", "chestx", "crop_disease"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=15)
    p.add_argument("--n_episodes", type=int, default=None, help="預設 1-shot=100、5-shot=400（CC-CDFSL 設定）")

    p.add_argument("--backbone", default="ViT-B/16")
    p.add_argument("--lora_encoder", default="both", choices=["vision", "text", "both"])
    p.add_argument("--lora_qkv_mode", default="separate", choices=["separate", "fused"])
    p.add_argument("--lora_out_proj", action="store_true", help="也在 out_proj 加 LoRA（CLIP-LoRA 預設不加）")
    p.add_argument("--lora_r", type=int, default=2)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.25)

    p.add_argument("--steps", type=int, default=500,
                   help="每個 episode 的梯度更新步數。5-way 下 support ≤ batch_size，1 epoch = 1 步，"
                        "因此論文的「100 epochs」若照字面只有 100 步，會嚴重訓練不足（見 reports/0924_Episodic協定對齊報告.md）")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--no_ce_aug", action="store_true", help="關閉 CE 訓練的 RandomResizedCrop+flip")

    p.add_argument("--lambda1", type=float, default=0.0, help="T-I-T 權重（0 ⇒ 不算）")
    p.add_argument("--lambda2", type=float, default=0.0, help="I-T-I 權重（0 ⇒ 不算）")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_aug", type=int, default=4)

    p.add_argument("--use_feature_sr", action="store_true")
    p.add_argument("--sr_scale", type=int, default=2)
    p.add_argument("--sr_refiner_layers", type=int, default=0)
    p.add_argument("--sr_refiner_heads", type=int, default=8)
    p.add_argument("--lambda3", type=float, default=0.3)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--save_dir", default="./results_episodic")
    p.add_argument("--tag", default=None, help="結果檔名後綴；預設依設定自動產生")
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CE_AUG = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
])


def ce_augment(images: torch.Tensor) -> torch.Tensor:
    return torch.stack([CE_AUG(img) for img in images])


def default_tag(args) -> str:
    if args.use_feature_sr:
        mode = f"fsr_s{args.sr_scale}_ref{args.sr_refiner_layers}_l3{args.lambda3:g}"
    else:
        mode = "nofsr"
    cc = f"cc_l1{args.lambda1:g}_l2{args.lambda2:g}" if (args.lambda1 > 0 or args.lambda2 > 0) else "nocc"
    return f"{mode}_{cc}_lora{args.lora_encoder}_r{args.lora_r}_steps{args.steps}"


def run_episode(episode, clip_lora, args, device, cc_loss_fn, cr_loss_fn, use_amp):
    support_images = episode["support_images"].to(device)
    support_labels = episode["support_labels"].to(device)
    query_images = episode["query_images"].to(device)
    query_labels = episode["query_labels"].to(device)
    class_names = episode["class_names"]

    clip_lora.reset_lora()
    feature_sr = None
    if args.use_feature_sr:
        feature_sr = FeatureSRModule(
            feat_dim=clip_lora.clip.output_dim, input_size=14, scale=args.sr_scale,
            refiner_layers=args.sr_refiner_layers, refiner_heads=args.sr_refiner_heads,
        ).to(device)

    params = list(clip_lora.trainable_parameters())
    if feature_sr is not None:
        params += list(feature_sr.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    n_support = support_images.shape[0]
    iters_per_epoch = math.ceil(n_support / args.batch_size)
    total_iters = args.steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_cc = args.lambda1 > 0 or args.lambda2 > 0
    logit_scale = clip_lora.clip.model.logit_scale.exp().detach()

    clip_lora.train()
    if feature_sr is not None:
        feature_sr.train()
    for step in range(total_iters):
        b = step % iters_per_epoch
        if b == 0:
            perm = torch.randperm(n_support, device=device)
        idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
        imgs, labels = support_images[idx], support_labels[idx]
        if not args.no_ce_aug:
            imgs = ce_augment(imgs)

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            text_feat = clip_lora.encode_text(class_names)
            cls, patches = clip_lora.encode_image_with_patches(imgs)
            loss_ce = F.cross_entropy(logit_scale * cls @ text_feat.T, labels)
            loss = loss_ce
            loss_tit = loss_iti = loss_cr = torch.zeros((), device=device)

            if feature_sr is not None:
                patches, patches_orig = feature_sr(patches, return_both=True)
                loss_cr = cr_loss_fn(patches, patches_orig)
                loss = loss + args.lambda3 * loss_cr

            if use_cc:
                with torch.no_grad():
                    aug_images = torch.cat([augment_tensor_batch(imgs) for _ in range(args.n_aug)], dim=0)
                    _, aug_patches = clip_lora.encode_image_with_patches(aug_images)
                    if feature_sr is not None:
                        aug_patches, _ = feature_sr(aug_patches, return_both=False)
                loss_tit, loss_iti = cc_loss_fn(
                    text_feat=text_feat, patch_feat_orig=patches, patch_feat_aug=aug_patches,
                )
                loss = loss + args.lambda1 * loss_tit + args.lambda2 * loss_iti

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    last = {"loss": loss.item(), "ce": loss_ce.item(), "tit": loss_tit.item(),
            "iti": loss_iti.item(), "cr": loss_cr.item()}

    clip_lora.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        text_feat = clip_lora.encode_text(class_names)
        q_cls, _ = clip_lora.encode_image_with_patches(query_images)
        acc = compute_few_shot_accuracy(q_cls.float() @ text_feat.float().T, query_labels)
    return acc, last


def main():
    args = parse_args()
    device = "cuda"
    n_episodes = args.n_episodes or (100 if args.n_shot == 1 else 400)
    tag = args.tag or default_tag(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = save_dir / f"{args.dataset}_{args.n_shot}shot_{tag}.jsonl"

    done = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["episode"]] = rec["acc"]
    logger.info(f"Episodic protocol | {args.dataset} {args.n_way}-way {args.n_shot}-shot | "
                f"{n_episodes} episodes × {args.steps} steps | tag={tag}")
    logger.info(f"Args: {vars(args)}")
    if done:
        logger.info(f"Resuming: {len(done)} episodes already in {out_jsonl}")

    pool = get_dataset(args.dataset, args.data_root, split="train",
                       transform=get_clip_transform(img_size=224), bscd_mode=True)
    logger.info(f"Episode pool (BSCD 全量): {pool}")
    sampler = FewShotEpisodeSampler(pool, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query)

    clip_wrapper = CLIPWrapper(backbone=args.backbone, device=device)
    clip_lora = CLIPLoRA(
        clip_wrapper, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        encoder=args.lora_encoder, qkv_mode=args.lora_qkv_mode, use_out_proj=args.lora_out_proj,
    )
    n_lora = sum(p.numel() for p in clip_lora.trainable_parameters())
    logger.info(f"LoRA trainable parameters: {n_lora:,} "
                f"(encoder={args.lora_encoder}, qkv={args.lora_qkv_mode}, out_proj={args.lora_out_proj})")
    cc_loss_fn = CyclicConsistencyLoss(top_k=args.top_k)
    cr_loss_fn = CrossResolutionConsistencyLoss(input_size=14, scale=args.sr_scale).to(device)
    use_amp = not args.no_amp

    accs = [done[e] for e in sorted(done)]
    t0 = time.time()
    n_new = 0
    with open(out_jsonl, "a") as f:
        for ep in range(n_episodes):
            if ep in done:
                continue
            seed_everything(args.seed * 100003 + ep)
            episode = sampler.sample()
            acc, last = run_episode(episode, clip_lora, args, device, cc_loss_fn, cr_loss_fn, use_amp)
            f.write(json.dumps({"episode": ep, "acc": acc, "classes": episode["class_names"], **last}) + "\n")
            f.flush()
            accs.append(acc)
            n_new += 1
            if n_new % args.log_every == 0 or ep == n_episodes - 1:
                mean, ci = compute_confidence_interval(accs) if len(accs) > 1 else (accs[0], 0.0)
                per_ep = (time.time() - t0) / n_new
                eta_h = per_ep * (n_episodes - len(accs)) / 3600
                logger.info(f"  [{len(accs)}/{n_episodes}] running {mean:.2f} ± {ci:.2f}% | "
                            f"last loss {last['loss']:.4f} (ce {last['ce']:.4f}) | "
                            f"{per_ep:.1f}s/episode, ETA {eta_h:.2f}h")

    mean, ci = compute_confidence_interval(accs)
    logger.info(f"Final (episodic, {tag}): {mean:.2f} ± {ci:.2f}% (n_episodes={len(accs)})")
    summary = {"dataset": args.dataset, "n_shot": args.n_shot, "tag": tag, "mean_acc": mean,
               "ci": ci, "n_episodes": len(accs), "args": vars(args)}
    (save_dir / f"{args.dataset}_{args.n_shot}shot_{tag}.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
