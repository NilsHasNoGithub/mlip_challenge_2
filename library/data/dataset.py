import io
from optparse import Option
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from ..config import MaskPos, TrainMetadata, ExpConfig
from .utils import list_index, paths_to_labels, read_img, apply_mask
import pytorch_lightning as pl
import random
import albumentations
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image as pil_img
import warnings
from ..models import _legacy_timm_model
from ..inference_utils import correct_img_rotation

class HotelDataSet(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        txt_labels: List[str],
        label_encoder: Dict[str, int],
        augmentation_pipeline: Optional[albumentations.Compose] = None,
        image_transforms: Optional[
            transforms.Compose
        ] = None,  # applied after augmentations
        mask_positions: Optional[List[MaskPos]] = None,
        include_file_name: bool = False,
        is_eval: bool = False,
        is_hotels_50k: bool = False,
        rot_model_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._img_paths = image_paths

        self._txt_labels = txt_labels
        self._mask_positions = mask_positions

        assert (
            mask_positions is None or len(mask_positions) > 0
        ), "mask positions can not be empty"

        self._include_file_name = include_file_name
        self._is_eval = is_eval
        self._is_hotels_50k = is_hotels_50k

        self._labels = (
            [label_encoder[l] for l in self._txt_labels]
            if not is_eval
            else [0 for _ in self._img_paths]
        )

        self._augmentation_pipeline = (
            augmentation_pipeline
            if augmentation_pipeline is not None
            else albumentations.Compose([])
        )
        self._image_transforms = (
            image_transforms if image_transforms is not None else transforms.Compose([])
        )

        self._rot_model = (
            _legacy_timm_model.TimmModule.load_from_checkpoint(
                rot_model_ckpt, pretrained=False, for_inference=True
            ).cpu().eval()
            if rot_model_ckpt is not None
            else None
        )

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, index: int) -> Any:
        img_path = self._img_paths[index]
        lbl = self._labels[index]

        img = read_img(img_path)

        if self._rot_model is not None:
            img = correct_img_rotation(self._rot_model, img)

        # TODO apply data augmentation (before mask)
        img = self._augmentation_pipeline(image=img)["image"]

        if self._mask_positions is not None:
            msk = random.choice(self._mask_positions)
            img = apply_mask(img, msk)

        img = self._image_transforms(pil_img.fromarray(img))

        if self._include_file_name:
            return img, lbl, os.path.split(img_path)[1]

        return img, lbl


class HotelLightningModule(pl.LightningDataModule):
    def __init__(
        self,
        train_metadata: TrainMetadata,
        exp_config: ExpConfig,
        num_dl_workers: int = 4,
        augmentation_pipeline: Optional[albumentations.Compose] = None,
        val_augmentation_pipeline: Optional[albumentations.Compose] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self._train_metadata = train_metadata
        self._exp_config = exp_config
        self._num_workers = num_dl_workers
        self._augmentation_pipeline = augmentation_pipeline
        self._val_augmentation_pipeline = val_augmentation_pipeline
        self._transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        md = self._train_metadata
        self._train_ds = HotelDataSet(
            list_index(md.images, md.train_idxs),
            list_index(md.txt_labels, md.train_idxs),
            md.label_encoder,
            mask_positions=md.mask_positions,
            augmentation_pipeline=self._augmentation_pipeline,
            image_transforms=self._transform,
        )
        self._val_ds = HotelDataSet(
            list_index(md.images, md.val_idxs),
            list_index(md.txt_labels, md.val_idxs),
            md.label_encoder,
            mask_positions=md.mask_positions,
            augmentation_pipeline=self._val_augmentation_pipeline,
            image_transforms=self._transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._exp_config.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self._exp_config.batch_size,
            num_workers=self._num_workers,
        )
