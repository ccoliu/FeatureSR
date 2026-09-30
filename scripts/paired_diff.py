"""
scripts/paired_diff.py
兩個 train_episodic.py 結果檔（.jsonl）的逐 episode 配對比較。

只比較兩邊都有的 episode（依 episode 編號配對），並檢查兩邊的類別是否一致，
確認真的是同一批 episode。判定「真實效果」的標準：配對差的 95% CI 不含 0。

用法：
  python scripts/paired_diff.py B.jsonl A.jsonl          # 回報 B − A
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import compute_confidence_interval


def load(path):
    recs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["episode"]] = r
    return recs


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    b, a = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        sys.exit(f"共同 episode 只有 {len(common)} 個，無法計算")
    mismatch = [e for e in common if a[e]["classes"] != b[e]["classes"]]
    if mismatch:
        sys.exit(f"episode {mismatch[:5]} 的類別不一致，兩個檔案不是同一批 episode")

    ma, ca = compute_confidence_interval([a[e]["acc"] for e in common])
    mb, cb = compute_confidence_interval([b[e]["acc"] for e in common])
    md, cd = compute_confidence_interval([b[e]["acc"] - a[e]["acc"] for e in common])
    print(f"n = {len(common)} 個共同 episode（A {len(a)}、B {len(b)}）")
    print(f"A  {ma:6.2f} ± {ca:.2f}   {Path(sys.argv[2]).name}")
    print(f"B  {mb:6.2f} ± {cb:.2f}   {Path(sys.argv[1]).name}")
    verdict = "CI 不含 0" if abs(md) > cd else "CI 含 0（雜訊內）"
    print(f"B − A  {md:+.2f} ± {cd:.2f}   {verdict}")


if __name__ == "__main__":
    main()
