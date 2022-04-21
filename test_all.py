import numpy as np
import pytest
from library.metric import mean_average_precision
from library.data.utils import apply_mask, get_train_img_paths, get_mask_img_paths
import random
import os
from library.exp_config import TrainMetadata
from library.data import dataset
import cv2

TRAIN_METADATA_FILE = "data/train_metadata.yml"


# def test_dataset():
#     train_metadata = TrainMetadata.from_yaml(TRAIN_METADATA_FILE)
#     ds = dataset.HotelDataSet(
#         train_metadata.train_imgs,
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
