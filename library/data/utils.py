from typing import Any, Dict, List, Tuple
import glob
import os
from dataclasses import dataclass
from importlib_metadata import metadata
import torch
import yaml
from collections import defaultdict
import numpy as np
from PIL import Image as pil_image
from ..exp_config import MaskPos


def paths_to_labels(paths: List[str]) -> List[str]:
    result = []

    for p in paths:
        parts = p.split("/")
        assert len(parts) >= 2

        result.append(parts[-2])

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
    return list(
        map(os.path.abspath, glob.glob(os.path.join(train_img_folder, "*/*.jpg")))
    )


def get_mask_img_paths(mask_img_folder: str) -> List[str]:
    return list(map(os.path.abspath, glob.glob(os.path.join(mask_img_folder, "*.png"))))


def apply_mask(
    img: torch.Tensor, mask: MaskPos, fill_value=(1.0, 0.0, 0.0)
) -> np.ndarray:
    """
    Overlay mask over 3d img
    """
    result = img.clone()

    l, t, w, h = mask.to_indices(img.shape[1], img.shape[0])

    for i in range(len(fill_value)):
        result[i, t : t + h, l : l + w] = fill_value[i]

    return result


def read_img(img_path: str) -> np.ndarray:
    img = np.array(pil_image.open(img_path).convert("RGB"))

    return img
