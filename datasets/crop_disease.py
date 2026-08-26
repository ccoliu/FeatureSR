"""
datasets/crop_disease.py
CropDiseases 資料集（植物病害，38 類）

下載：https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
結構：
    <root>/CropDiseases/dataset/
        Apple___Apple_scab/
        Apple___Black_rot/
        ...（38 個類別資料夾）
"""
from pathlib import Path
from .base_dataset import FewShotDataset


class CropDiseaseDataset(FewShotDataset):
    """
    Plant Village CropDiseases 資料集。
    包含 38 個植物病害類別，每類約 1,000~2,000 張影像。
    """

    NAME = "CropDiseases"

    def _load_data(self):
        root = self.root / "CropDiseases"

        # 嘗試多種可能的路徑結構（依序搜尋）
        candidate_dirs = [
            # 本程式碼預設結構
            root / "dataset",
            # Kaggle 下載解壓後的結構（常見）
            root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train",
            root / "New Plant Diseases Dataset(Augmented)" / "train",
            root / "train",
        ]

        dataset_dir = None
        for d in candidate_dirs:
            if d.exists() and any(d.iterdir()):
                dataset_dir = d
                break

        if dataset_dir is None:
            raise FileNotFoundError(
                f"CropDiseases dataset not found under {root}.\n"
                "請確認資料集已解壓至以下任一結構：\n"
                f"  {root}/dataset/<class_name>/\n"
                f"  {root}/New Plant Diseases Dataset(Augmented)/.../train/<class_name>/\n"
                "Download: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"
            )

        print(f"[CropDiseases] 使用路徑：{dataset_dir}")
        self._build_from_folder(dataset_dir, split_ratio=0.8)

