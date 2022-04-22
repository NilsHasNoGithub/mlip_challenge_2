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

def make_predictions(
    test_dir: str, inference_config_file: str, train_metadata_file: str, out_file: str, num_workers=multiprocessing.cpu_count(), device = None,
):
    config: InferenceConfig = InferenceConfig.from_yaml_file(inference_config_file)
    train_metadata = TrainMetadata.from_yaml(train_metadata_file)

    # model = TimmModule.load_from_checkpoint(config.model_path, pretrained=False)
    model = torch.load(config.model_path)
    # with open(config.model_path, "rb") as f:
    #     model = pickle.load(f)

    img_paths = glob.glob(os.path.join(test_dir, "*.jpg"), recursive=True)

    dataset = HotelDataSet(
        # img_paths,
        train_metadata.val_imgs,
        train_metadata.label_encoder,
        augmentation_pipeline=augmentations.VAL_PRESETS[config.val_augmentation_preset],
        image_transforms=model.get_transform(),
        # is_eval=True,
        include_file_name=True
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.eval()
    model = model.to(device)

    result = defaultdict(list)
    labels = []
    all_preds = []

    i2 = 0
    for inputs, lbl, paths in tqdm(DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers)):
        inputs = inputs.to(device)
        predictions = model.forward(inputs)

        
        for i in range(predictions.shape[0]):
            top_5 = torch.topk(predictions[i, :], 5, largest=True).indices.cpu()
            # top_5_np = np.argsort(predictions[i, :].cpu().detach().numpy())[-5:][::-1]
            # print(predictions[i, :][top_5])
            result["image_id"].append(paths[i])
            top_5_hotel_ids = []
            for j in range(top_5.shape[0]):
                top_5_hotel_ids.append(train_metadata.label_decoder[top_5[j].item()])

            result["hotel_id"].append(" ".join(top_5_hotel_ids))
            all_preds.append(predictions[i, :].detach().cpu())
            labels.append(lbl[i].detach().cpu())

        if i2 > 10:
            break
        i2 += 1

    all_preds = torch.stack(all_preds)
    labels = torch.stack(labels)

    print(f"map5: {mean_average_precision(all_preds.numpy(), labels.numpy())}")

    result_df = pd.DataFrame.from_dict(result)
    result_df.to_csv(out_file, index=False)


@click.command()
@click.option("--test-dir", "-d", type=click.Path(exists=True))
@click.option("--config-file", "-c", type=click.Path(exists=True))
@click.option("--train-metadata", "-t", type=click.Path(exists=True))
@click.option("--out-file", "-o", type=click.Path())
@click.option("--num-workers", "-j", type=int, default=multiprocessing.cpu_count()//2)
@click.option("--device", type=str, default=None)
def main(
    test_dir: str,
    config_file: str,
    train_metadata: str,
    out_file: str,
    num_workers: int,
    device: Optional[str]
):
    make_predictions(test_dir, config_file, train_metadata, out_file, num_workers=num_workers, device=device)


if __name__ == "__main__":
    main()
