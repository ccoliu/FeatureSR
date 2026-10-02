"""
scripts/within_model_diff.py
同一個結果檔內兩個欄位的逐 episode 配對比較（同一個模型、同一批 query），例如推論時上採樣的診斷：

  python scripts/within_model_diff.py results.jsonl acc_any28 acc        # 回報 acc_any28 − acc
  python scripts/within_model_diff.py results.jsonl                     # 列出所有 acc_* 欄位對 acc 的差
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import compute_confidence_interval


def main():
    if len(sys.argv) not in (2, 4):
        sys.exit(__doc__)
    recs = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    pairs = [(sys.argv[2], sys.argv[3])] if len(sys.argv) == 4 else \
            [(k, "acc") for k in recs[0] if k.startswith("acc_")]
    print(f"n = {len(recs)} 個 episode  {Path(sys.argv[1]).name}")
    for b, a in pairs:
        mb, _ = compute_confidence_interval([r[b] for r in recs])
        md, cd = compute_confidence_interval([r[b] - r[a] for r in recs])
        verdict = "CI 不含 0" if abs(md) > cd else "雜訊內"
        print(f"  {b:18s} {mb:6.2f}   − {a:5s} {md:+.2f} ± {cd:.2f}   {verdict}")


if __name__ == "__main__":
    main()
