"""
datasets/base_dataset.py
所有資料集的基底類別
"""
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image


class FewShotDataset(Dataset, ABC):
    """
    Few-Shot 資料集基底類別。
    子類別需實作 _load_data() 和 class_names 屬性。
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        n_shot: Optional[int] = None,
        bscd_mode: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.n_shot = n_shot
        self.bscd_mode = bscd_mode  # BSCD-FSL 官方協定：全量資料進 episodic pool，無 train/test split

        # 由子類別實作：填充 self.data 和 self.label2idx
        self.data: List[Tuple[str, int]] = []   # (image_path, class_idx)
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}
        self.class_names: List[str] = []

        self._load_data()

        # 如果是 train 且指定了 n_shot，限制每個類別的樣本數
        if self.split == "train" and self.n_shot is not None:
            self._keep_few_shot_only()

    def _keep_few_shot_only(self):
        class_to_data = {}
        for img_path, label in self.data:
            if label not in class_to_data:
                class_to_data[label] = []
            class_to_data[label].append((img_path, label))
        
        few_shot_data = []
        import random
        # 使用固定 seed 確保每次選出的 few-shot 訓練影像固定
        state = random.getstate()
        random.seed(42)
        for label, items in class_to_data.items():
            selected = random.sample(items, min(self.n_shot, len(items)))
            few_shot_data.extend(selected)
        random.setstate(state)
        self.data = few_shot_data
        print(f"[{self.__class__.__name__}] Strict few-shot training enabled. Total training samples: {len(self.data)} (n_shot={self.n_shot})")

    @abstractmethod
    def _load_data(self):
        """載入資料，填充 self.data, self.label2idx, self.idx2label, self.class_names"""
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        # Windows 長路徑處理：\\?\ 前綴需搭配絕對路徑
        open_path = img_path
        if os.name == 'nt':
            abs_path = os.path.abspath(img_path).replace('/', '\\')
            if len(abs_path) > 255:
                open_path = '\\\\?\\' + abs_path
        img = Image.open(open_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_class_samples(self, class_idx: int) -> List[int]:
        """取得特定類別的所有樣本索引"""
        return [i for i, (_, label) in enumerate(self.data) if label == class_idx]

    def get_few_shot_task(
        self,
        n_way: int,
        n_shot: int,
        n_query: int,
        class_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        從資料集中取樣一個 few-shot 任務。

        Returns:
            support_indices: support set 的樣本索引
            query_indices: query set 的樣本索引
            selected_classes: 選出的類別索引
        """
        n_classes = len(self.class_names)
        if class_indices is None:
            selected_classes = random.sample(range(n_classes), n_way)
        else:
            selected_classes = class_indices

        support_indices = []
        query_indices = []

        for cls in selected_classes:
            samples = self.get_class_samples(cls)
            assert len(samples) >= n_shot + n_query, (
                f"Class {cls} has only {len(samples)} samples, "
                f"need at least {n_shot + n_query}"
            )
            chosen = random.sample(samples, n_shot + n_query)
            support_indices.extend(chosen[:n_shot])
            query_indices.extend(chosen[n_shot:])

        return support_indices, query_indices, selected_classes

    def _build_from_folder(self, folder: Path, split_ratio: float = 0.8):
        """
        從資料夾結構建立資料清單（子資料夾名稱 = 類別名）。

        Args:
            folder: 包含各類別子資料夾的根目錄
            split_ratio: 訓練集比例（bscd_mode=True 時忽略，使用全部資料）
        """
        classes = sorted([d.name for d in folder.iterdir() if d.is_dir()])
        self.class_names = classes
        self.label2idx = {c: i for i, c in enumerate(classes)}
        self.idx2label = {i: c for c, i in self.label2idx.items()}

        all_data = []
        for cls in classes:
            cls_dir = folder / cls
            imgs = []
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    imgs.append(str(p))
            imgs.sort()
            label = self.label2idx[cls]
            all_data.extend([(img, label) for img in imgs])

        if self.bscd_mode:
            # BSCD-FSL 官方協定：無 held-out split，全量資料用於 episodic pool
            random.seed(42)
            random.shuffle(all_data)
            random.seed()
            self.data = all_data
        else:
            # 原始 80/20 隨機分割
            random.seed(42)
            random.shuffle(all_data)
            split_idx = int(len(all_data) * split_ratio)
            if self.split == "train":
                self.data = all_data[:split_idx]
            else:
                self.data = all_data[split_idx:]
            random.seed()  # 重置 seed

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split}, "
            f"n_classes={len(self.class_names)}, "
            f"n_samples={len(self.data)})"
        )
