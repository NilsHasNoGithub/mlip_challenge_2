import io
from optparse import Option
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from ..exp_config import MaskPos, TrainMetadata, ExpConfig
from .utils import paths_to_labels, read_img, apply_mask
import pytorch_lightning as pl
import random
import albumentations
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image as pil_img


class HotelDataSet(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        label_encoder: Dict[str, int],
        augmentation_pipeline: Optional[albumentations.Compose] = None,
        image_transforms: Optional[
            transforms.Compose
        ] = None,  # applied after augmentations
        mask_positions: Optional[List[MaskPos]] = None,
    ) -> None:
        super().__init__()

        self._img_paths = image_paths

        self._txt_labels = paths_to_labels(image_paths)
        self._mask_positions = mask_positions

        self._labels = [label_encoder[l] for l in self._txt_labels]

        self._augmentation_pipeline = (
            augmentation_pipeline
            if augmentation_pipeline is not None
            else albumentations.Compose([])
        )
        self._image_transforms = (
            image_transforms if image_transforms is not None else transforms.Compose([])
        )

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, index: int) -> Any:
        img_path = self._img_paths[index]
        lbl = self._labels[index]

        img = read_img(img_path)

        # TODO apply data augmentation (before mask)
        img = self._augmentation_pipeline(image=img)["image"]
        img = self._image_transforms(pil_img.fromarray(img))

        if self._mask_positions is not None:
            msk = random.choice(self._mask_positions)
            img = apply_mask(img, msk)

        return img, lbl


class HotelLightningModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_metadata: TrainMetadata,
        exp_config: ExpConfig,
        num_dl_workers: int = 4,
        augmentation_pipeline: Optional[albumentations.Compose] = None,
        val_augmentation_pipeline: Optional[albumentations.Compose] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self._train_metadata = train_metadata
        self._data_dir = data_dir
        self._exp_config = exp_config
        self._num_workers = num_dl_workers
        self._augmentation_pipeline = augmentation_pipeline
        self._val_augmentation_pipeline = val_augmentation_pipeline
        self._transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        md = self._train_metadata
        self._train_ds = HotelDataSet(
            md.train_imgs,
            md.label_encoder,
            mask_positions=md.mask_positions,
            augmentation_pipeline=self._augmentation_pipeline,
            image_transforms=self._transform,
        )
        self._val_ds = HotelDataSet(
            md.val_imgs,
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
