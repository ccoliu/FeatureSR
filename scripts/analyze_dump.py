"""
scripts/analyze_dump.py
離線比較 ATHA baseline 上各種「推論時」的訊號（scripts/run_atha_dump.sh 存下的 ep####.npz）。
模型與訓練完全是 baseline；這裡只改變推論時怎麼打分，所以所有比較都是同一個模型、同一批 query 的配對。

訊號（每個都是 [Q, C] 的分數）：
  cls                      q_cls · t_coop，即 baseline 本身（會檢查與模型實際輸出一致）
  txt_{coop,hand}_{val,plain}
                           patch 對文字：每類取 top-k（10%）patch 相似度的平均
                           coop = CoOp 學到的 prompt；hand = 固定 prompt「a photo of a {name}.」
  dn4_{val,plain}_{all,top}
                           DN4 式 image-to-class（Li et al., CVPR 2019）：每個 query patch 對該類所有 support patch
                           取最大相似度，再對 query patch 取平均（all）或取前 10% 的平均（top）
  proto                    q_cls · 每類 support CLS 的平均（全域 support 原型）。對照組：用來區分增益是來自
                           「局部比對」還是只是「用了 support 影像」
融合（只看 argmax，所以只有相對權重有意義）：
  raw   cls + w · x              w ∈ {0.25, 0.5, 1, 2}
  z     z(cls) + w · z(x)        z = 每個 query 在類別間標準化；w ∈ {0.25, 0.5, 1}

所有候選在這裡事先固定。流程：在開發集（seed 2）上列出全部結果、依「配對差平均最大」選一個設定，
再用 --only 在確認集（seed 1，400 個 episode）上只評估那一個設定。

用法：
  python scripts/analyze_dump.py ~/atha_dumps/ISIC_5shot_seed2
  python scripts/analyze_dump.py ~/atha_dumps/ISIC_5shot_seed1 --only dn4_val_top:z:0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import compute_confidence_interval

K_FRAC = 0.1
RAW_W = (0.25, 0.5, 1.0, 2.0)
Z_W = (0.25, 0.5, 1.0)


def topk_mean(x, k_frac, dim):
    k = max(1, round(k_frac * x.shape[dim]))
    return x.topk(k, dim=dim).values.mean(dim=dim)


def signals(d, device):
    t = lambda k: torch.from_numpy(d[k]).to(device)
    q_cls, s_cls = t("q_cls").float(), t("s_cls").float()
    t_coop, t_hand = t("t_coop").float(), t("t_hand").float()
    s_y = torch.from_numpy(d["s_y"]).to(device)
    C = t_coop.shape[0]
    out = {"cls": q_cls @ t_coop.T}

    for pname in ("val", "plain"):
        qp = t(f"q_{pname}")                                      # [Q, P, d] fp16
        sp = t(f"s_{pname}")                                      # [S, P, d] fp16
        for tname, tf in (("coop", t_coop), ("hand", t_hand)):
            sim = torch.einsum("qpd,cd->qcp", qp.float(), tf)     # [Q, C, P]
            out[f"txt_{tname}_{pname}"] = topk_mean(sim, K_FRAC, dim=-1)
        # DN4：support 依類別排序（s_y = repeat(range(C), n_shot)），每類的 patch 攤平成一個描述子集合
        assert torch.equal(s_y, s_y.sort().values)
        Q, P, D = qp.shape
        sim = (qp.reshape(-1, D) @ sp.reshape(-1, D).T).float()   # [Q·P, S·P]
        best = sim.reshape(Q, P, C, -1).max(dim=-1).values         # [Q, P, C]：每個 query patch 對每類的最近鄰
        out[f"dn4_{pname}_all"] = best.mean(dim=1)
        out[f"dn4_{pname}_top"] = topk_mean(best, K_FRAC, dim=1)

    proto = F.normalize(torch.stack([s_cls[s_y == c].mean(0) for c in range(C)]), dim=-1)
    out["proto"] = q_cls @ proto.T
    return out


def zscore(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


def configs(sig):
    """{(訊號, 融合, 權重): 分數}；('cls', '-', 0) 是 baseline"""
    cls = sig["cls"]
    yield ("cls", "-", 0.0), cls
    for name, x in sig.items():
        if name == "cls":
            continue
        yield (name, "alone", 0.0), x
        for w in RAW_W:
            yield (name, "raw", w), cls + w * x
        for w in Z_W:
            yield (name, "z", w), zscore(cls) + w * zscore(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--only", default=None, help="只評估一個設定，格式 訊號:融合:權重，例如 dn4_val_top:z:0.5")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    only = None
    if args.only:
        n, f, w = args.only.split(":")
        only = (n, f, float(w))

    files = sorted(Path(args.dump_dir).expanduser().glob("ep*.npz"))
    if not files:
        sys.exit(f"{args.dump_dir} 裡沒有 ep*.npz")
    accs = {}
    agree = []
    for fpath in files:
        d = np.load(fpath)
        q_y = torch.from_numpy(d["q_y"]).to(device)
        with torch.no_grad():
            sig = signals(d, device)
        agree.append((sig["cls"].argmax(1).cpu().numpy() == d["q_logits"].argmax(1)).mean())
        for key, score in configs(sig):
            if only and key not in (("cls", "-", 0.0), only):
                continue
            accs.setdefault(key, []).append((score.argmax(1) == q_y).float().mean().item() * 100)

    print(f"{len(files)} 個 episode  {args.dump_dir}")
    print(f"cls 重算預測與模型輸出一致的比例：{np.mean(agree):.4f}（應為 1）")
    base = np.array(accs[("cls", "-", 0.0)])
    print(f"baseline（cls）：{base.mean():.2f}\n")
    print(f"{'訊號':22s} {'融合':6s} {'權重':>5s} {'準確率':>7s}   {'− baseline（配對）':>20s}")
    rows = []
    for key, a in accs.items():
        if key == ("cls", "-", 0.0):
            continue
        a = np.array(a)
        md, cd = compute_confidence_interval(list(a - base))
        rows.append((key, a.mean(), md, cd))
    for (name, fusion, w), m, md, cd in rows:
        mark = " *" if abs(md) > cd else ""
        wstr = f"{w:g}" if fusion in ("raw", "z") else ""
        print(f"{name:22s} {fusion:6s} {wstr:>5s} {m:7.2f}   {md:+7.2f} ± {cd:.2f}{mark}")
    fused = [r for r in rows if r[0][1] in ("raw", "z")]
    if fused and not only:
        best = max(fused, key=lambda r: r[2])
        (name, fusion, w), m, md, cd = best
        print(f"\n配對差平均最大的融合設定：{name}:{fusion}:{w:g}（{md:+.2f} ± {cd:.2f}）")
    print("\n* = 配對 CI 不含 0")


if __name__ == "__main__":
    main()
