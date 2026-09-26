"""
make_atha_json.py
把本專案的四個 BSCD 資料集轉成 ATHA 官方程式碼（github.com/shuaiyi308/ATHA）需要的
cdfsl-benchmark 格式：{out_dir}/{Name}/novel.json = {"image_names": [...], "image_labels": [...]}。

標籤整數沿用本專案的類別順序；已驗證四個資料集的類別順序與 ATHA coop_lora_trainer.py 內
寫死的 label_names 一致（ATHA 用完整名稱，例如 ISIC 的 "Melanoma"，本專案原本用縮寫 "MEL"）。
路徑一律寫成絕對路徑，因此必須在實際要跑 ATHA 的機器上執行。

用法：python scripts/make_atha_json.py --out_dir ~/atha_data
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets import get_dataset

NAME_MAP = {"crop_disease": "CropDiseases", "eurosat": "EuroSAT", "isic": "ISIC", "chestx": "ChestX"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    for ours, name in NAME_MAP.items():
        ds = get_dataset(ours, args.data_root, split="train", transform=None, bscd_mode=True)
        names = [str(Path(path).resolve()) for path, _ in ds.data]
        labels = [int(label) for _, label in ds.data]
        missing = [n for n in names[:50] if not Path(n).exists()]
        if missing:
            raise FileNotFoundError(f"{name}: 找不到影像，例如 {missing[0]}")
        out = Path(args.out_dir).expanduser() / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "novel.json").write_text(json.dumps({"image_names": names, "image_labels": labels}))
        counts = Counter(labels)
        print(f"{name}: {len(names)} 張, {len(counts)} 類, 每類最少 {min(counts.values())} 張 → {out / 'novel.json'}")


if __name__ == "__main__":
    main()
