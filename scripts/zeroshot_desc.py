"""
scripts/zeroshot_desc.py
類別描述的品質檢查：原始 CLIP ViT-B/16（不微調）的 zero-shot 準確率，比較三種文字：
  name：a photo of a {名稱}.
  cat ：a photo of a {名稱}, {描述}.
  only：a photo of a {描述}.
（與 ATHA 的 CoOp 初始 prompt「a photo of a」相同；名稱為 ATHA 的 label_names，底線換成空白）
影像：每類最多取 --per_class 張（固定亂數）；報告全類別準確率，以及 5-way 15-query episode 的平均（同 BSCD 協定）。

用法：python scripts/zeroshot_desc.py [--datasets ISIC ChestX EuroSAT CropDiseases]
"""
import argparse
import json
import re
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["ISIC", "ChestX", "EuroSAT", "CropDiseases"])
ap.add_argument("--per_class", type=int, default=100)
ap.add_argument("--episodes", type=int, default=2000)
args = ap.parse_args()

FSR = Path(__file__).resolve().parent.parent
DESC = json.load(open(FSR / "configs/class_descriptions.json"))
ROOTS = {"CropDiseases": Path.home() / "atha_data_bscd"}   # CropDiseases 用 BSCD 標準版
SRC = open(Path.home() / "ATHA/coop_lora_trainer_local.py").read()

dev = "cuda"
model, preprocess = clip.load("ViT-B/16", device=dev)
model.eval()


@torch.no_grad()
def text_feats(prompts):
    t = model.encode_text(clip.tokenize(prompts).to(dev)).float()
    return t / t.norm(dim=-1, keepdim=True)


@torch.no_grad()
def image_feats(paths, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
        f = model.encode_image(x).float()
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out)


for ds in args.datasets:
    m = re.search(r'dataset == "%s":\s*\n\s*label_names=\[(.*?)\]' % ds, SRC, re.S)
    names = re.findall(r'"([^"]*)"', m.group(1))
    meta = json.load(open(ROOTS.get(ds, Path.home() / "atha_data") / ds / "novel.json"))
    paths, labels = np.array(meta["image_names"]), np.array(meta["image_labels"])
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.permutation(np.where(labels == c)[0])[:args.per_class] for c in range(len(names))])
    x = image_feats(list(paths[keep]))
    y = torch.tensor(labels[keep], device=dev)

    texts = {
        "name": ["a photo of a %s." % n.replace("_", " ") for n in names],
        "cat": ["a photo of a %s, %s." % (n.replace("_", " "), DESC[ds][n]) for n in names],
        "only": ["a photo of a %s." % DESC[ds][n] for n in names],
    }
    # 固定的 5-way 15-query episode（三種文字共用）
    erng = np.random.default_rng(1)
    eps = []
    for _ in range(args.episodes):
        cls = erng.choice(len(names), 5, replace=False)
        q = np.concatenate([erng.choice(np.where(labels[keep] == c)[0], 15, replace=False) for c in cls])
        eps.append((cls, q))
    res = {}
    for k, t in texts.items():
        sim = x @ text_feats(t).t()
        full = (sim.argmax(1) == y).float().mean().item() * 100
        ep_acc = []
        yy = y.cpu().numpy()
        s = sim.cpu().numpy()
        for cls, q in eps:
            pred = cls[s[q][:, cls].argmax(1)]
            ep_acc.append((pred == yy[q]).mean() * 100)
        ep_acc = np.array(ep_acc)
        res[k] = (full, ep_acc)
    print(f"== {ds}（{len(names)} 類，{len(keep)} 張）")
    for k, (full, ep) in res.items():
        line = f"  {k:<5} 全類別 {full:6.2f}   5-way {ep.mean():6.2f}"
        if k != "name":
            d = ep - res["name"][1]
            line += f"   − name {d.mean():+6.2f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.2f}"
        print(line, flush=True)
