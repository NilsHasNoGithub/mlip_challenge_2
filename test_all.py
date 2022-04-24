import numpy as np
import pytest
from library.metric import mean_average_precision
from library.data.utils import apply_mask, get_train_img_paths, get_mask_img_paths
import random
import os
from library.config import TrainMetadata
from library.data import dataset
from library.layers.pooling import GlobalAveragePool2D
import torch

TRAIN_METADATA_FILE = "data/train_metadata.yml"


def test_ga_pool():
    i = torch.randn((2, 3, 4, 5))
    assert tuple(GlobalAveragePool2D()(i).shape) == (2, 3)
    assert tuple(GlobalAveragePool2D(start_dim=1, end_dim=2)(i).shape) == (2, 5)
    assert tuple(GlobalAveragePool2D(start_dim=-3, end_dim=-2)(i).shape) == (2, 5)

# def test_dataset():
#     train_metadata = TrainMetadata.from_yaml(TRAIN_METADATA_FILE)
#     ds = dataset.HotelDataSet(
#         train_metadata.images[train_metadata.train_idxs],
#         train_metadata.label_encoder,
#         augmentation_pipeline=None,
#         mask_positions=train_metadata.mask_positions,
#     )

#     i = 0
#     for img, lbl in ds:
#         i += 1
#         if i > 10:
#             break


def test_map():
    pred = np.array([[1, 0, 0, 0, 0, 0]])
    lbl = [0]

    print(mean_average_precision(pred, lbl))
