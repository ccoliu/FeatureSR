"""
utils/metrics.py
評估指標工具函數
"""
import math
from typing import Tuple, List
import numpy as np
import torch


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """計算分類準確率"""
    correct = (predictions == targets).sum().item()
    total = targets.shape[0]
    return correct / total * 100.0


def compute_confidence_interval(
    accs: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """
    計算均值和 95% 信賴區間（使用 t-distribution）。

    Args:
        accs: 準確率列表（percentage，如 84.5）
        confidence: 信賴水準（預設 0.95）

    Returns:
        mean: 平均準確率
        ci:   ± 信賴區間
    """
    n = len(accs)
    mean = np.mean(accs)
    std = np.std(accs, ddof=1)

    # t-score for 95% CI
    if confidence == 0.95:
        t_score = 1.96
    elif confidence == 0.99:
        t_score = 2.576
    else:
        from scipy import stats
        t_score = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    ci = t_score * std / math.sqrt(n)
    return float(mean), float(ci)


def compute_few_shot_accuracy(
    logits: torch.Tensor,     # [Q, K]
    query_labels: torch.Tensor,  # [Q]
) -> float:
    """
    計算 few-shot 任務的 query set 準確率。

    Args:
        logits:       [Q, K] 分類 logits 或相似度
        query_labels: [Q]   ground truth labels（0-indexed）

    Returns:
        accuracy: 百分比
    """
    preds = logits.argmax(dim=-1)
    return compute_accuracy(preds, query_labels)
