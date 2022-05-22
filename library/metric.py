from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn


def mean_average_precision(logits: np.ndarray, labels, k=5) -> float:
    total = 0
    for i in range(logits.shape[0]):
        top_k = np.argsort(logits[i])[-k:][::-1]

        for j in range(k):
            if top_k[j] == labels[i]:
                total += 1 / (j + 1)
                break

    return total / logits.shape[0]

    # result.append(np.mean(np.take(labels[i], top_5)))


def mean_average_precision_topk(
    top_k_preds: List[List[int]], labels: List[int], k=5
) -> float:
    total = 0

    for top_k in top_k_preds:
        assert len(top_k) >= k
        for i in range(k):
            if top_k[i] == labels[i]:
                total += 1 / (i + 1)

    return total / len(top_k_preds)
