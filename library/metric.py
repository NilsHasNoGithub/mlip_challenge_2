import numpy as np


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
