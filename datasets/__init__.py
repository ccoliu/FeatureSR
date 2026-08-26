"""
datasets/__init__.py
資料集模組統一入口
"""
from .crop_disease import CropDiseaseDataset
from .eurosat import EuroSATDataset
from .isic import ISICDataset
from .chestx import ChestXDataset

DATASET_REGISTRY = {
    "crop_disease": CropDiseaseDataset,
    "eurosat": EuroSATDataset,
    "isic": ISICDataset,
    "chestx": ChestXDataset,
}


def get_dataset(name: str, root: str, split: str = "train", transform=None, **kwargs):
    """
    取得指定資料集。

    Args:
        name: 資料集名稱，見 DATASET_REGISTRY
        root: 資料根目錄
        split: 'train' / 'val' / 'test'
        transform: torchvision transforms
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](root=root, split=split, transform=transform, **kwargs)
