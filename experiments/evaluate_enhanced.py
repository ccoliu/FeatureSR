"""
experiments/evaluate_enhanced.py
驗證 Method 1 (領域 Multi-Prompt 集成) 與 Method 4 (測試期多視角增強 TTA) 的增益效果
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import get_dataset
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
from utils.few_shot_sampler import FewShotEpisodeSampler
from utils.augmentation import get_clip_transform
from utils.metrics import compute_few_shot_accuracy, compute_confidence_interval
from utils.prompt_templates import get_domain_text_embeddings

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_modes(
    dataset_name: str,
    checkpoint_path: str,
    n_shot: int,
    n_episodes: int,
    data_root: str = "./data",
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    在同一批 Episode 上對比 4 種推論模式：
      1. Standard (原始單一通用 Prompt, 無 TTA)
      2. + Prompt Ensemble (領域多模板集成, 無 TTA)
      3. + TTA (原始通用 Prompt + 測試期多視角)
      4. + Prompt Ensemble & TTA (兩者結合 ⭐)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = get_clip_transform(img_size=224)
    test_dataset = get_dataset(dataset_name, data_root, split="test", transform=transform)
    sampler = FewShotEpisodeSampler(
        dataset=test_dataset,
        n_way=5,
        n_shot=n_shot,
        n_query=15,
    )

    # 載入模型權重
    clip_wrapper = CLIPWrapper(backbone="ViT-B/16", device=device)
    clip_lora = CLIPLoRA(
        clip_wrapper=clip_wrapper,
        r=4,
        alpha=1.0,
        dropout=0.0,
    )
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "lora_state_dict" in ckpt:
            state = ckpt["lora_state_dict"]
        elif "clip_lora_state" in ckpt:
            state = ckpt["clip_lora_state"]
        else:
            state = ckpt
        clip_lora.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint ({len(state)} keys): {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using initial weights.")

    clip_lora.eval()

    accs_standard = []
    accs_prompt = []
    accs_tta = []
    accs_both = []

    for ep in range(n_episodes):
        episode = sampler.sample()
        query_images = episode["query_images"].to(device)
        query_labels = episode["query_labels"].to(device)
        class_names  = episode["class_names"]

        # --- 1. 文字特徵提取 ---
        text_feat_std = clip_lora.encode_text(class_names)
        text_feat_ens = get_domain_text_embeddings(
            clip_model=clip_lora.clip,
            class_names=class_names,
            dataset_name=dataset_name,
            device=device,
            use_ensemble=True,
        )

        # --- 2. 影像特徵提取 ---
        # (a) 標準視角
        q_cls_std, _ = clip_lora.encode_image_with_patches(query_images)
        q_cls_std = F.normalize(q_cls_std, dim=-1)

        # (b) TTA (原圖 + 水平翻轉高效融合)
        q_flip = torch.flip(query_images, dims=[-1])
        q_cls_flip, _ = clip_lora.encode_image_with_patches(q_flip)
        q_cls_flip = F.normalize(q_cls_flip, dim=-1)
        q_cls_tta = F.normalize((q_cls_std + q_cls_flip) / 2.0, dim=-1)

        # --- 3. 計算 4 種模式之準確率 ---
        logits_std = q_cls_std @ text_feat_std.T
        accs_standard.append(compute_few_shot_accuracy(logits_std, query_labels))

        logits_prompt = q_cls_std @ text_feat_ens.T
        accs_prompt.append(compute_few_shot_accuracy(logits_prompt, query_labels))

        logits_tta = q_cls_tta @ text_feat_std.T
        accs_tta.append(compute_few_shot_accuracy(logits_tta, query_labels))

        logits_both = q_cls_tta @ text_feat_ens.T
        accs_both.append(compute_few_shot_accuracy(logits_both, query_labels))

    m_std, ci_std = compute_confidence_interval(accs_standard)
    m_p, ci_p = compute_confidence_interval(accs_prompt)
    m_t, ci_t = compute_confidence_interval(accs_tta)
    m_b, ci_b = compute_confidence_interval(accs_both)

    return {
        "Standard": {"mean": m_std, "ci": ci_std},
        "Prompt_Ensemble": {"mean": m_p, "ci": ci_p, "delta": m_p - m_std},
        "TTA": {"mean": m_t, "ci": ci_t, "delta": m_t - m_std},
        "Prompt_and_TTA": {"mean": m_b, "ci": ci_b, "delta": m_b - m_std},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["eurosat", "crop_disease", "isic", "chestx", "all"])
    parser.add_argument("--n_shot", type=int, default=5, choices=[1, 5])
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--use_fsr_weights", action="store_true",
                        help="使用 Feature-SR 訓練的最佳 checkpoint (否則使用 Baseline checkpoint)")
    args = parser.parse_args()

    n_ep = args.n_episodes or (400 if args.n_shot == 5 else 100)
    datasets = ["eurosat", "crop_disease", "isic", "chestx"] if args.dataset == "all" else [args.dataset]
    weight_type = "Feature-SR (FSR)" if args.use_fsr_weights else "Baseline (BL)"

    print("=" * 85, flush=True)
    print(f" 🚀 Method 1 (Prompt Ensemble) + Method 4 (TTA) 強化評估驗證", flush=True)
    print(f" 測試設定: {args.n_shot}-shot | 測試 Episodes: {n_ep} | 評估模型: {weight_type}", flush=True)
    print("=" * 85, flush=True)

    summary_rows = []

    for ds in datasets:
        suffix = "_fsr_best.pth" if args.use_fsr_weights else "_best.pth"
        ckpt_path = f"./checkpoints/{ds}_{args.n_shot}shot{suffix}"
        
        logger.info(f"--- Testing Dataset: {ds.upper()} ({args.n_shot}-shot, {n_ep} episodes) ---")
        res = evaluate_modes(
            dataset_name=ds,
            checkpoint_path=ckpt_path,
            n_shot=args.n_shot,
            n_episodes=n_ep,
            data_root=args.data_root,
        )

        std_str = f"{res['Standard']['mean']:.2f} ± {res['Standard']['ci']:.2f}%"
        p_str = f"{res['Prompt_Ensemble']['mean']:.2f}% ({res['Prompt_Ensemble']['delta']:+.2f}%)"
        t_str = f"{res['TTA']['mean']:.2f}% ({res['TTA']['delta']:+.2f}%)"
        both_str = f"{res['Prompt_and_TTA']['mean']:.2f} ± {res['Prompt_and_TTA']['ci']:.2f}% ({res['Prompt_and_TTA']['delta']:+.2f}%)"

        print(f"\n[{ds.upper()}] 結果對比 (5-way {args.n_shot}-shot):", flush=True)
        print(f"  1. Standard (原始單一通用 Prompt):       {std_str}", flush=True)
        print(f"  2. + Prompt Ensemble (領域多模板 M1):    {p_str}", flush=True)
        print(f"  3. + Test-Time Augmentation (TTA M4):    {t_str}", flush=True)
        print(f"  4. ⭐ 結合 M1 + M4 (Prompt + TTA):        {both_str}", flush=True)

        summary_rows.append({
            "dataset": ds,
            "std": res['Standard']['mean'],
            "prompt": res['Prompt_Ensemble']['mean'],
            "tta": res['TTA']['mean'],
            "both": res['Prompt_and_TTA']['mean'],
            "delta": res['Prompt_and_TTA']['delta'],
        })

    avg_std = np.mean([r["std"] for r in summary_rows])
    avg_p = np.mean([r["prompt"] for r in summary_rows])
    avg_t = np.mean([r["tta"] for r in summary_rows])
    avg_both = np.mean([r["both"] for r in summary_rows])
    avg_delta = avg_both - avg_std

    print("\n" + "=" * 85, flush=True)
    print(f" 🏆 4 大跨域資料集平均表現 (Average Across 4 Datasets):", flush=True)
    print(f"  - 原始 Standard 平均:               {avg_std:.2f}%", flush=True)
    print(f"  - + Prompt Ensemble (M1) 平均:       {avg_p:.2f}% ({avg_p-avg_std:+.2f}%)", flush=True)
    print(f"  - + TTA (M4) 平均:                   {avg_t:.2f}% ({avg_t-avg_std:+.2f}%)", flush=True)
    print(f"  - ⭐ 兩者結合 (M1 + M4) 最終平均:     {avg_both:.2f}% ({avg_delta:+.2f}%) 🚀", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
