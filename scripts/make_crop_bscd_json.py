"""
scripts/make_crop_bscd_json.py
BSCD-FSL 原始 benchmark 的 CropDiseases（Kaggle saroz014/plant-disease，壓縮檔內的 dataset/train/，43,456 張、38 類）
轉成 ATHA 的清單格式。標籤 = 類別資料夾名稱排序後的索引（已驗證與 ATHA coop_lora_trainer 的 label_names 順序完全一致）。
輸出到獨立的資料根目錄（ATHA 依資料夾名稱 CropDiseases 決定類別名稱，因此不能改名）：
  ~/atha_data_bscd/CropDiseases/novel.json   →  訓練時用 -data_path ~/atha_data_bscd

用法：python scripts/make_crop_bscd_json.py [--src data/CropDiseases_bscd/dataset/train]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(Path(__file__).resolve().parent.parent / "data/CropDiseases_bscd/dataset/train"))
args = ap.parse_args()

src = Path(args.src)
classes = sorted(p.name for p in src.iterdir() if p.is_dir())
names, labels = [], []
for label, c in enumerate(classes):
    for f in sorted((src / c).iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            names.append(str(f.resolve()))
            labels.append(label)
dst = Path.home() / "atha_data_bscd/CropDiseases/novel.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({"image_names": names, "image_labels": labels}))
counts = Counter(labels)
print(f"{len(names)} 張、{len(classes)} 類；每類最少 {min(counts.values())}、最多 {max(counts.values())} → {dst}")
