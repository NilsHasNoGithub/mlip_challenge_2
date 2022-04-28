import click
from library.models.timm_model import TimmModule
from library.config import TrainMetadata
from library.data.dataset import HotelDataSet
import library.data.augmentations as augs
import torch
import numpy as np
import pandas as pd


def generate_embeddings(
    train_metadata: TrainMetadata, model: TimmModule, batch_size=4, n_epochs=10
) -> pd.DataFrame:
    ds = HotelDataSet(
        train_metadata.images,
        train_metadata.label_encoder,
        augs.PRESETS["default"],
        model.get_transform(),
        mask_positions=train_metadata.mask_positions,
    )


@click.command()
@click.option(
    "--data-dir", "-d", type=click.Path(exists=True), help="path to train imgs"
)
@click.option(
    "--train-metadata",
    "-t",
    type=click.Path(exists=True),
    help="Path to train metadata file",
)
@click.command(
    "--model-path", "-m", type=click.Path(exists=True), help="path to pretrained model"
)
def main(
    data_dir,
    train_metadata,
):
    train_metadata = TrainMetadata.from_yaml(train_metadata)
    model = TimmModule(train_metadata)
    ...
