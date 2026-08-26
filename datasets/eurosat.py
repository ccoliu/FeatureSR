"""
datasets/eurosat.py
EuroSAT 資料集（衛星影像土地覆蓋分類，10 類）

下載：https://github.com/phelber/EuroSAT
結構：
    <root>/EuroSAT/2750/
        AnnualCrop/
        Forest/
        HerbaceousVegetation/
        Highway/
        Industrial/
        Pasture/
        PermanentCrop/
        Residential/
        River/
        SeaLake/
"""
from pathlib import Path
from .base_dataset import FewShotDataset


class EuroSATDataset(FewShotDataset):
    """
    EuroSAT 衛星影像資料集。
    包含 10 個土地利用類別，每類 2,000~3,000 張 64×64 RGB 影像。
    """

    NAME = "EuroSAT"

    CLASS_NAMES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake",
    ]

    def _load_data(self):
        dataset_dir = self.root / "EuroSAT" / "2750"
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"EuroSAT dataset not found at {dataset_dir}.\n"
                "Please download from: https://madm.dfki.de/files/sentinel/EuroSAT.zip\n"
                "Or run: python download_datasets.py --dataset eurosat --data_root <root>"
            )
        self._build_from_folder(dataset_dir, split_ratio=0.8)
