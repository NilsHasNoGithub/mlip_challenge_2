from typing import Any, Dict, Iterable, List, Tuple
import glob
import os
from dataclasses import dataclass
from importlib_metadata import metadata
import torch
import yaml
from collections import defaultdict
import numpy as np
from PIL import Image as pil_image
from ..config import MaskPos


def paths_to_labels(paths: List[str], hotels_50k=False) -> List[str]:
    result = []

    for p in paths:
        parts = p.split("/")
        idx = 3 if hotels_50k else 2
        assert len(parts) >= idx

        result.append(parts[-idx])

    return result


def make_label_map(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    c = 0
    encoder = {}

    for l in labels:
        if l in encoder.keys():
            continue

        encoder[l] = c
        c += 1

    return encoder, {v: k for (k, v) in encoder.items()}


def class_counts(labels) -> Dict[Any, int]:
    cls_counts = defaultdict(int)

    for l in labels:
        cls_counts[l] += 1

    return dict(**cls_counts)


def get_train_img_paths(train_img_folder: str) -> List[str]:
    return list(glob.glob(os.path.join(train_img_folder, "**/*.jpg"), recursive=True))


def get_mask_img_paths(mask_img_folder: str) -> List[str]:
    return list(glob.glob(os.path.join(mask_img_folder, "*.png")))


def apply_mask(
    img: np.ndarray, mask: MaskPos, fill_value=(255, 0.0, 0.0)
) -> np.ndarray:
    """
    Overlay mask over 3d img
    """
    # result = img.clone()
    result = img.copy()

    l, t, w, h = mask.to_indices(img.shape[1], img.shape[0])

    for i in range(len(fill_value)):
        result[t : t + h, l : l + w, i] = fill_value[i]

    return result


def read_img(img_path: str) -> np.ndarray:
    img = np.array(pil_image.open(img_path).convert("RGB"))

    return img


def list_index(l: list, idxs: Iterable) -> list:
    return [l[i] for i in idxs]


def read_img_rot(img_path: str, rotation: int) -> np.ndarray:
    img = np.array(
        pil_image.open(img_path).convert("RGB").rotate(rotation * 90, expand=True)
    )

    return img
