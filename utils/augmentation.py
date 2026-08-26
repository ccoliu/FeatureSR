"""
utils/augmentation.py
影像變換設定

CLIP 標準預處理 + Semantic Anchor augmentation
"""
from typing import Callable
import torchvision.transforms as T


def get_clip_transform(img_size: int = 224) -> Callable:
    """
    CLIP 標準預處理 transform（用於 support/query 影像）。

    Returns:
        torchvision transform compose
    """
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])


def get_aug_transform(img_size: int = 224) -> Callable:
    """
    Semantic Anchor Module 使用的 augmentation transform。

    論文未詳述具體的 augmentation 策略，此處參考常見 few-shot 設定：
    - 隨機水平翻轉
    - 隨機旋轉 ±15°
    - 輕微顏色抖動
    - 保持與 CLIP preprocess 相同的 resize + normalize

    Returns:
        torchvision transform compose
    """
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomApply([
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        ], p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ], p=0.3),
        T.ToTensor(),
        normalize,
    ])
