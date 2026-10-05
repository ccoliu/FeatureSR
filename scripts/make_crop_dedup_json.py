"""
scripts/make_crop_dedup_json.py
從 ATHA 的 CropDiseases 清單（Kaggle「New Plant Diseases Dataset (Augmented)」的 train，70,295 張）濾掉離線增強副本，
只保留原圖，避免同一張原圖的增強副本同時出現在一個 episode（會特別放大 transductive 方法的增益）。
增強副本可由檔名後綴辨認：_flipLR、_flipTB、_90deg、_180deg、_270deg、_new{N}degFlip{LR,TB}。
輸出到獨立的資料根目錄（ATHA 依資料夾名稱 CropDiseases 決定類別名稱，因此不能改名），標籤不變：
  ~/atha_data_dedup/CropDiseases/novel.json   →  訓練時用 -data_path ~/atha_data_dedup
"""
import json
import re
from collections import Counter
from pathlib import Path

AUG = re.compile(r"_(flip(LR|TB)|180deg|90deg|270deg|new[0-9]+degFlip(LR|TB))\.[A-Za-z]+$")
src = Path.home() / "atha_data/CropDiseases/novel.json"
dst = Path.home() / "atha_data_dedup/CropDiseases/novel.json"
d = json.loads(src.read_text())
keep = [i for i, n in enumerate(d["image_names"]) if not AUG.search(n)]
out = {k: [d[k][i] for i in keep] for k in ("image_names", "image_labels")}
for k in d:
    if k not in out:
        out[k] = d[k]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out))
per_class = Counter(out["image_labels"])
print(f"原清單 {len(d['image_names'])} 張 → 去除增強副本後 {len(keep)} 張，{len(per_class)} 類；"
      f"每類最少 {min(per_class.values())}、最多 {max(per_class.values())}")
print(f"寫入 {dst}")
