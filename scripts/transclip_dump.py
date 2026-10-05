"""
scripts/transclip_dump.py
在 ATHA baseline 存下的特徵（scripts/run_atha_dump.sh 的 ep####.npz）上，逐 episode 離線套用 TransCLIP-FS
（Zanella et al., NeurIPS 2024；官方程式 github.com/MaxZanella/transduction-for-vlms，clone 在 ~/transduction-for-vlms）。
Transductive：同時利用該 episode 的 75 個 query（不用它們的標註）與 25 張 support 推論。模型與訓練完全是 baseline。

與官方 TransCLIP_solver 的差異只有一處：官方依序跑 gamma ∈ {0.002, 0.01, 0.02, 0.2}（狀態延續），再用另外切出的
有標註驗證樣本挑 gamma；episode 協定沒有額外標註資料，因此這裡記下每個 gamma 階段結束時的預測，
每個 (初始化, gamma) 都是事先固定的候選，在開發集（seed 2）上選定、再到確認集（seed 1）只評估選定的那一個。

候選（共 8 個）：
  初始化  clip100：官方預設，y_hat = softmax(100 · q·t)
          model4 / model2：模型自己的校準，y_hat = softmax(s · q·t)，s 為 ATHA 的 logit scale（5-shot 4、1-shot 2）
  gamma   0.002 / 0.01 / 0.02 / 0.2（官方的 few-shot 候選）

用法：
  python scripts/transclip_dump.py ~/atha_dumps/ISIC_5shot_seed2
  python scripts/transclip_dump.py ~/atha_dumps/ISIC_5shot_seed1 --only model4:0.01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path.home() / "transduction-for-vlms"))   # 放最後：該 repo 根目錄的 utils.py 會蓋掉本專案的 utils 套件
from utils.metrics import compute_confidence_interval
from TransCLIP_solver.TransCLIP_utils import (Gaussian, build_affinity_matrix, init_mu, init_sigma, init_z,
                                              update_mu, update_sigma, update_z)

GAMMAS = (0.002, 0.01, 0.02, 0.2)      # 官方 get_parameters 的 few-shot 候選
LAMBDA, MAX_ITER, N_NEIGHBORS = 0.5, 10, 3


def transclip_fs(q, s, s_y, t, init, model_scale=4):
    """回傳 {gamma: 預測}；流程同官方 TransCLIP_solver（gamma 依序跑、狀態延續）"""
    K, d = t.shape
    n = q.shape[0]
    if init == "clip100":
        y_hat, z = init_z(100 * q @ t.T, softmax=True)
    else:
        y_hat, z = init_z(F.softmax(model_scale * q @ t.T, dim=1), softmax=False)
    adapter = Gaussian(mu=init_mu(K, d, z, q, s, s_y), std=init_sigma(d, 1 / d)).cuda()
    W = build_affinity_matrix(q, s, n, N_NEIGHBORS)
    preds = {}
    for gamma in GAMMAS:
        for k in range(MAX_ITER + 1):
            z = update_z(adapter(q, no_exp=True), y_hat, z, W, LAMBDA, N_NEIGHBORS, s_y)[0:n]
            if k == MAX_ITER:
                break
            adapter = update_mu(adapter, gamma, q, z, s, s_y)
            adapter = update_sigma(adapter, gamma, q, z, s, s_y)
        preds[gamma] = z.argmax(1)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--only", default=None, help="只評估一個設定，格式 初始化:gamma，例如 model4:0.01")
    args = ap.parse_args()

    files = sorted(Path(args.dump_dir).expanduser().glob("ep*.npz"))
    if not files:
        sys.exit(f"{args.dump_dir} 裡沒有 ep*.npz")
    accs = {}
    base = []
    for fpath in files:
        d = np.load(fpath)
        g = lambda k: torch.from_numpy(d[k]).cuda()
        q, s, t = F.normalize(g("q_cls").float(), dim=-1), F.normalize(g("s_cls").float(), dim=-1), g("t_coop").float()
        q_y, s_y = g("q_y"), g("s_y")
        base.append((torch.from_numpy(d["q_logits"]).cuda().argmax(1) == q_y).float().mean().item() * 100)
        with torch.no_grad():
            scale = 4 if len(s_y) // t.shape[0] == 5 else 2          # ATHA：5-shot 4、1-shot 2
            for init in ("clip100", "model"):
                for gamma, pred in transclip_fs(q, s, s_y, t, init, scale).items():
                    key = f"{init}{scale if init == 'model' else ''}:{gamma:g}"
                    if args.only and key != args.only:
                        continue
                    accs.setdefault(key, []).append((pred == q_y).float().mean().item() * 100)

    base = np.array(base)
    print(f"{len(files)} 個 episode  {args.dump_dir}")
    print(f"baseline（inductive，模型輸出）：{base.mean():.2f}\n")
    print(f"{'設定（初始化:gamma）':22s} {'準確率':>7s}   {'− baseline（配對）':>20s}")
    rows = []
    for key, a in accs.items():
        a = np.array(a)
        md, cd = compute_confidence_interval(list(a - base))
        rows.append((key, a.mean(), md, cd))
        print(f"{key:22s} {a.mean():7.2f}   {md:+7.2f} ± {cd:.2f}{' *' if abs(md) > cd else ''}")
    if not args.only:
        key, m, md, cd = max(rows, key=lambda r: r[2])
        print(f"\n配對差平均最大的設定：{key}（{md:+.2f} ± {cd:.2f}）")
    print("\n* = 配對 CI 不含 0")


if __name__ == "__main__":
    main()
