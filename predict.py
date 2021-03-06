from collections import defaultdict
from email.policy import default
import multiprocessing
import os
from typing import Optional

import numpy as np
from library.models.timm_model import TimmModule
from library.config import InferenceConfig, TrainMetadata
from library.data.dataset import HotelDataSet
from library.data import augmentations
from library.metric import mean_average_precision
from torch.utils.data.dataloader import DataLoader
import glob
import torch
import click
import pandas as pd
from tqdm import tqdm
from icecream import ic
import pickle
import itertools


def make_predictions(
    test_dir: str,
    inference_config_file: str,
    train_metadata_file: str,
    out_file: str,
    num_workers=multiprocessing.cpu_count(),
    device=None,
):
    config: InferenceConfig = InferenceConfig.from_yaml_file(inference_config_file)
    train_metadata = TrainMetadata.from_yaml(train_metadata_file)

    model = TimmModule.load_from_checkpoint(
        config.model_path,
        pretrained=False,
        pretrained_timm_model=None,
        for_inference=True,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval()
    model: TimmModule = model.to(device)

    img_paths = glob.glob(os.path.join(test_dir, "**/*.jpg"), recursive=True)

    test_dataset = HotelDataSet(
        img_paths,
        None,
        None,
        augmentation_pipeline=augmentations.VAL_PRESETS[config.val_augmentation_preset],
        image_transforms=model.get_transform(),
        is_eval=True,
        include_file_name=True,
        rot_model_ckpt=config.rot_model_ckpt,
    )

    if config.embedding_based:
        train_dataset = HotelDataSet(
            train_metadata.images,
            train_metadata.txt_labels,
            train_metadata.label_encoder,
            augmentation_pipeline=augmentations.VAL_PRESETS[
                config.val_augmentation_preset
            ],
            image_transforms=model.get_transform(),
            rot_model_ckpt=config.rot_model_ckpt if config.embedding_rot_cor else None,
        )

        model.create_embeddings(
            DataLoader(
                train_dataset, batch_size=config.batch_size, num_workers=num_workers
            )
        )

    result = defaultdict(list)
    # labels = []
    # all_preds = []

    if config.embedding_based:
        embedding_db = model.embedding_db()
        lbl_db = model.label_db()

    for inputs, lbl, paths in tqdm(
        DataLoader(test_dataset, batch_size=config.batch_size, num_workers=num_workers)
    ):
        inputs = inputs.to(device)

        if config.embedding_based:
            top_ks = model.predict_based_on_embedding(
                inputs, embedding_db=embedding_db, lbl_db=lbl_db
            )

            for i, top_5 in enumerate(top_ks):
                result["image_id"].append(paths[i])

                top_5_hotel_ids = []
                for j in range(len(top_5)):
                    top_5_hotel_ids.append(train_metadata.label_decoder[top_5[j]])

                result["hotel_id"].append(" ".join(top_5_hotel_ids))

        else:
            predictions = model.forward(inputs)
            for i in range(predictions.shape[0]):
                top_5 = torch.topk(predictions[i, :], 5, largest=True).indices.cpu()
                # print(predictions[i, :][top_5])
                result["image_id"].append(paths[i])
                top_5_hotel_ids = []
                for j in range(top_5.shape[0]):
                    top_5_hotel_ids.append(
                        train_metadata.label_decoder[top_5[j].item()]
                    )

                result["hotel_id"].append(" ".join(top_5_hotel_ids))

    # all_preds = torch.stack(all_preds)
    # labels = torch.stack(labels)

    # print(f"map5: {mean_average_precision(all_preds.numpy(), labels.numpy())}")

    result_df = pd.DataFrame.from_dict(result)
    result_df.to_csv(out_file, index=False)


@click.command()
@click.option("--test-dir", "-d", type=click.Path(exists=True))
@click.option("--config-file", "-c", type=click.Path(exists=True))
@click.option("--train-metadata", "-t", type=click.Path(exists=True))
@click.option("--out-file", "-o", type=click.Path())
@click.option("--num-workers", "-j", type=int, default=multiprocessing.cpu_count() // 2)
@click.option("--device", type=str, default=None)
def main(
    test_dir: str,
    config_file: str,
    train_metadata: str,
    out_file: str,
    num_workers: int,
    device: Optional[str],
):
    make_predictions(
        test_dir,
        config_file,
        train_metadata,
        out_file,
        num_workers=num_workers,
        device=device,
    )


if __name__ == "__main__":
    main()
