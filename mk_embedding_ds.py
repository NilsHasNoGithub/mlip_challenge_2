from collections import defaultdict
import copy
import os
import random
from typing import List, Optional, Tuple
import click
from tqdm import tqdm
from library.data.utils import list_index
from library.models.timm_model import TimmModule
from library.config import TrainMetadata
from library.data.dataset import HotelDataSet
import library.data.augmentations as augs
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.model_selection import train_test_split


def generate_embeddings(
    new_train_metadata: TrainMetadata,
    model: TimmModule,
    batch_size=4,
    n_epochs=10,
    device="cpu",
    num_workers=6,
    no_split=False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    md = new_train_metadata

    model = model.eval()
    model = model.to(device)

    train_idxs = md.train_idxs + md.val_idxs if no_split else md.train_idxs

    train_ds = HotelDataSet(
        list_index(md.images, train_idxs),
        list_index(md.txt_labels, train_idxs),
        md.label_encoder,
        augmentation_pipeline=augs.PRESETS["fgvc8_winner"],
        image_transforms=model.get_transform(),
        mask_positions=md.mask_positions,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)

    train_feats = []
    train_lbls = []

    for _ in tqdm(range(n_epochs), desc="Generating train data"):

        for imgs, lbls in tqdm(train_dl, leave=False):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            train_lbls.extend(lbls.tolist())

            for i in range(feats.shape[0]):
                train_feats.append(feats[i, :].detach().cpu().numpy())

    train_feats = np.stack(train_feats)
    train_labels = np.array(train_lbls)

    if no_split:
        return train_feats, train_labels, None, None

    val_ds = HotelDataSet(
        list_index(md.images, md.val_idxs),
        list_index(md.txt_labels, md.val_idxs),
        md.label_encoder,
        augmentation_pipeline=augs.VAL_PRESETS["none"],
        image_transforms=model.get_transform(),
        mask_positions=md.mask_positions,
    )

    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    val_feats = []
    val_lbls = []

    for imgs, lbls in tqdm(val_dl, "Generating validation data"):
        imgs = imgs.to(device)
        feats = model.forward_features(imgs)
        val_lbls.extend(lbls.tolist())

        for i in range(feats.shape[0]):
            val_feats.append(feats[i, :].detach().cpu().numpy())

    val_feats = np.stack(val_feats)
    val_labels = np.array(val_lbls)

    return train_feats, train_labels, val_feats, val_labels


def create_new_split(train_idxs, val_idxs, labels) -> Tuple[List, List]:
    """
    Create new split with all data from val in train
    """

    train_idxs = set(train_idxs)
    val_idxs = set(val_idxs)

    all_idxs = list(range(len(labels)))

    new_train, new_val = train_test_split(all_idxs, stratify=labels)

    new_train = set(new_train)
    new_val = set(new_val)

    train_idxs_per_label = defaultdict(list)

    for idx in new_train:
        if idx not in val_idxs:
            train_idxs_per_label[labels[idx]].append(idx)

    for l in train_idxs_per_label.values():
        random.shuffle(l)

    for idx in new_val:
        if idx in val_idxs:
            lbl = labels[idx]
            try:
                swap_idx = train_idxs_per_label[lbl].pop()
            except IndexError:
                continue

            new_train.remove(swap_idx)
            new_train.add(idx)

            new_val.remove(idx)
            new_val.add(swap_idx)

    # new_val_idxs = {i for i in random.sample(train_idxs, len(val_idxs))}
    # new_train_idxs = [i for i in train_idxs if i not in new_val_idxs] + val_idxs

    return [i for i in new_train], [i for i in new_val]


@click.command()
@click.option(
    "--train-metadata",
    "-t",
    type=click.Path(exists=True),
    help="Path to train metadata file",
)
@click.option(
    "--output-folder",
    "-o",
    type=click.Path(),
    help="path to store created datasets for training meta model",
)
@click.option(
    "--n-epochs",
    "-n",
    type=int,
    default=10,
    help="amount of training epochs to generate data for",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    help="Batch size to use for inference (only affects performance)",
)
@click.option(
    "--model-path", "-m", type=click.Path(exists=True), help="path to pretrained model"
)
@click.option(
    "--device", "-d", type=str, default="cpu", help="device to use for training"
)
@click.option(
    "--num-dl-workers",
    "-w",
    type=int,
    default=6,
    help="number of workers for data loader",
)
@click.option(
    "--no-split",
    is_flag=True,
    default=False,
    help="Use all data for generating embeddings",
)
def main(
    train_metadata,
    output_folder: str,
    n_epochs: int,
    batch_size: int,
    model_path: str,
    device: str,
    num_dl_workers: int,
    no_split: bool,
):
    os.makedirs(output_folder, exist_ok=True)

    train_metadata = TrainMetadata.from_yaml(train_metadata)

    if not no_split:
        new_train_idxs, new_val_idxs = create_new_split(
            train_metadata.train_idxs,
            train_metadata.val_idxs,
            train_metadata.txt_labels,
        )

        new_train_metadata = copy.deepcopy(train_metadata)
        new_train_metadata.train_idxs = new_train_idxs
        new_train_metadata.val_idxs = new_val_idxs
    else:
        new_train_metadata = train_metadata

    model = TimmModule.load_from_checkpoint(
        model_path, pretrained=False, pretrained_timm_model=None
    )
    full_data = generate_embeddings(
        new_train_metadata,
        model,
        device=device,
        num_workers=num_dl_workers,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    assert len(full_data) == 4 or (no_split and len(full_data) == 2)
    for data, name in zip(
        full_data, ["train_feats", "train_labels", "val_feats", "val_labels"]
    ):
        filepath = os.path.join(output_folder, f"{name}.npy")
        np.save(filepath, data, allow_pickle=False)


if __name__ == "__main__":
    main()
