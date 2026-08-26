"""
datasets/chestx.py
ChestX-ray8 胸部X光資料集（7 類，論文設定）

下載：https://nihcc.app.box.com/v/ChestXray-NIHCC
結構：
    <root>/ChestX/
        images/          # .png 影像
        Data_Entry_2017.csv
"""
import os
import csv
import random
from pathlib import Path
from typing import List, Tuple
from .base_dataset import FewShotDataset


class ChestXDataset(FewShotDataset):
    """
    NIH ChestX-ray8 資料集，按照 CDFSL benchmark 選取 7 個類別。

    論文使用的 7 個類別（依照 CDFSL benchmark）：
        Atelectasis, Cardiomegaly, Effusion, Infiltration,
        Mass, Nodule, Pneumonia
    """

    NAME = "ChestX"

    # BSCD-FSL benchmark 使用的 7 個類別
    # 參考： https://github.com/IBM/cdfsl-benchmark/blob/master/datasets/Chest_few_shot.py
    # used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    #                "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
    # labels_maps 遭排 Pneumonia，实際用 7 分類 (0~6):
    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumothorax",   # 官方協定是 Pneumothorax，不是 Pneumonia
    ]

    def _load_data(self):
        img_dir = self.root / "ChestX-ray8" / "images" / "images"
        csv_file = self.root / "ChestX-ray8" / "Data_Entry_2017_v2020.csv"

        if not img_dir.exists() or not csv_file.exists():
            raise FileNotFoundError(
                f"ChestX dataset not found.\n"
                f"Expected image dir: {img_dir}\n"
                f"Expected CSV: {csv_file}\n"
                "Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
                "Note: Extract all images_0XX.tar.gz into images/ and place Data_Entry_2017.csv in ChestX-ray8/\n"
                "Or run: python download_datasets.py --dataset chestx --data_root <root>"
            )

        self.class_names = self.CLASS_NAMES
        self.label2idx = {c: i for i, c in enumerate(self.CLASS_NAMES)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}

        target_classes = set(self.CLASS_NAMES)
        all_data: List[Tuple[str, int]] = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Image Index"]
                finding = row["Finding Labels"]

                # 只取單一類別的樣本（排除多標籤）
                labels = [l.strip() for l in finding.split("|")]
                matching = [l for l in labels if l in target_classes]

                if len(matching) == 1:  # 嚴格單標籤
                    label = self.label2idx[matching[0]]
                    img_path = str(img_dir / img_name)
                    if os.path.exists(img_path):
                        all_data.append((img_path, label))

        if self.bscd_mode:
            # BSCD-FSL 官方協定：全量資料進 episodic pool
            random.seed(42)
            random.shuffle(all_data)
            random.seed()
            self.data = all_data
        else:
            # 原始 80/20 分割
            random.seed(42)
            random.shuffle(all_data)
            split_idx = int(len(all_data) * 0.8)
            if self.split == "train":
                self.data = all_data[:split_idx]
            else:
                self.data = all_data[split_idx:]
            random.seed()
