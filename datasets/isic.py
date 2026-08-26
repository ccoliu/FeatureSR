"""
datasets/isic.py
ISIC2018 皮膚病變資料集（7 類）

下載：https://challenge2018.isic-archive.com/
結構：
    <root>/ISIC2018/
        ISIC2018_Task3_Training_Input/       # 影像 .jpg
        ISIC2018_Task3_Training_GroundTruth/ # ISIC2018_Task3_Training_GroundTruth.csv
"""
import os
import csv
from pathlib import Path
from typing import List, Tuple
from .base_dataset import FewShotDataset


class ISICDataset(FewShotDataset):
    """
    ISIC 2018 Task 3 皮膚病變分類資料集。
    包含 7 個類別，共約 10,000 張影像。

    類別：
        MEL (Melanoma)
        NV (Melanocytic nevi)
        BCC (Basal cell carcinoma)
        AKIEC (Actinic keratoses)
        BKL (Benign keratosis-like lesions)
        DF (Dermatofibroma)
        VASC (Vascular lesions)
    """

    NAME = "ISIC2018"

    CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def _load_data(self):
        img_dir = self.root / "ISIC2018" / "ISIC2018_Task3_Training_Input"
        gt_file = (
            self.root / "ISIC2018"
            / "ISIC2018_Task3_Training_GroundTruth"
            / "ISIC2018_Task3_Training_GroundTruth.csv"
        )

        if not img_dir.exists() or not gt_file.exists():
            raise FileNotFoundError(
                f"ISIC2018 dataset not found.\n"
                f"Expected image dir: {img_dir}\n"
                f"Expected GT file: {gt_file}\n"
                "Download from: https://challenge2018.isic-archive.com/\n"
                "Or run: python download_datasets.py --dataset isic --data_root <root>"
            )

        # 建立類別映射
        self.class_names = self.CLASS_NAMES
        self.label2idx = {c: i for i, c in enumerate(self.CLASS_NAMES)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}

        # 解析 CSV
        all_data: List[Tuple[str, int]] = []
        with open(gt_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"] + ".jpg"
                img_path = str(img_dir / img_name)
                if not os.path.exists(img_path):
                    continue
                # one-hot 格式：找值為 '1.0' 或 '1' 的欄位
                label = None
                for cls in self.CLASS_NAMES:
                    if cls in row and float(row[cls]) > 0.5:
                        label = self.label2idx[cls]
                        break
                if label is not None:
                    all_data.append((img_path, label))

        if self.bscd_mode:
            # BSCD-FSL 官方協定：全量資料進 episodic pool
            import random
            random.seed(42)
            random.shuffle(all_data)
            random.seed()
            self.data = all_data
        else:
            # 原始 80/20 分割
            import random
            random.seed(42)
            random.shuffle(all_data)
            split_idx = int(len(all_data) * 0.8)
            if self.split == "train":
                self.data = all_data[:split_idx]
            else:
                self.data = all_data[split_idx:]
            random.seed()
