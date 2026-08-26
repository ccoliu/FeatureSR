"""
utils/few_shot_sampler.py
Few-Shot Episode 取樣器

對應論文的 K-way N-shot 設定：
  - K = n_way（類別數）
  - N = n_shot（每類 support 樣本數）
  - Q = n_query（每類 query 樣本數）
"""
import random
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, Sampler


class FewShotEpisodeSampler:
    """
    Few-shot Episode 取樣器。

    每次 sample() 返回一個 episode：
    - support set: K * N 張影像
    - query set:   K * Q 張影像

    Args:
        dataset:  FewShotDataset 實例
        n_way:    類別數 K
        n_shot:   每類 support 樣本數 N
        n_query:  每類 query 樣本數 Q
    """

    def __init__(
        self,
        dataset,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        query_dataset = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.query_dataset = query_dataset if query_dataset is not None else dataset

        # 預先建立類別 → 樣本索引的映射（對於 support dataset）
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(dataset.data):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        # 預先建立類別 → 樣本索引的映射（對於 query dataset）
        self.query_class_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(self.query_dataset.data):
            if label not in self.query_class_to_indices:
                self.query_class_to_indices[label] = []
            self.query_class_to_indices[label].append(idx)

        # 過濾掉樣本數不足的類別
        self.valid_classes = [
            cls for cls, indices in self.class_to_indices.items()
            if len(indices) >= n_shot
        ]

        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes: need {n_way} but only {len(self.valid_classes)} "
                f"classes have >= {n_shot} samples."
            )

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        取樣一個 few-shot episode。

        Returns:
            {
                'support_images':  [K*N, 3, H, W]
                'support_labels':  [K*N]  0-indexed
                'query_images':    [K*Q, 3, H, W]
                'query_labels':    [K*Q]  0-indexed
                'class_indices':   [K]    原始資料集類別索引
                'class_names':     [K]    類別名稱
            }
        """
        # 隨機選 K 個類別
        selected_classes = random.sample(self.valid_classes, self.n_way)
        local_label_map = {cls: i for i, cls in enumerate(selected_classes)}

        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []

        for cls in selected_classes:
            local_label = local_label_map[cls]
            
            # 從 dataset (support set) 採樣
            indices = self.class_to_indices[cls]
            chosen_support = random.sample(indices, self.n_shot)
            for idx in chosen_support:
                img, _ = self.dataset[idx]
                support_imgs.append(img)
                support_labels.append(local_label)

            # 從 query_dataset (query set) 採樣
            query_indices = self.query_class_to_indices[cls]
            # 排除已被選作 support 的樣本
            support_paths = [self.dataset.data[i][0] for i in chosen_support]
            available_query = [
                idx for idx in query_indices 
                if self.query_dataset.data[idx][0] not in support_paths
            ]
            chosen_query = random.sample(available_query, self.n_query)
            for idx in chosen_query:
                img, _ = self.query_dataset[idx]
                query_imgs.append(img)
                query_labels.append(local_label)

        # 取得類別名稱
        class_names = [self.dataset.idx2label[cls] for cls in selected_classes]

        return {
            "support_images": torch.stack(support_imgs),     # [K*N, C, H, W]
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_images":   torch.stack(query_imgs),       # [K*Q, C, H, W]
            "query_labels":   torch.tensor(query_labels, dtype=torch.long),
            "class_indices":  torch.tensor(selected_classes, dtype=torch.long),
            "class_names":    class_names,
        }

    def __repr__(self) -> str:
        return (
            f"FewShotEpisodeSampler("
            f"n_way={self.n_way}, n_shot={self.n_shot}, n_query={self.n_query}, "
            f"n_valid_classes={len(self.valid_classes)})"
        )


def collate_few_shot_episode(episodes: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    將多個 episode 組合成 batch（若需要批次訓練）。
    通常 few-shot 訓練每次處理一個 episode，此函數備用。
    """
    keys = episodes[0].keys()
    batch = {}
    for k in keys:
        if k == "class_names":
            batch[k] = [ep[k] for ep in episodes]
        else:
            batch[k] = torch.stack([ep[k] for ep in episodes])
    return batch
