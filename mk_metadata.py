from collections import defaultdict
import copy
from email.policy import default
from genericpath import exists
import math
import random
from typing import Dict, List, Optional, Union
from joblib import Parallel, delayed
from tqdm import tqdm
import yaml
import click
from library.data import utils
from library.config import MaskPos, TrainMetadata
import os
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass
from PIL import Image as pil_img
from dataclasses import dataclass


@dataclass
class DsResults:
    images: List[str]
    labels: List[str]
    label_encoder: Dict[str, int]
    label_decoder: Dict[int, str]
    train_idxs: List[int]
    val_idxs: List[int]


def extr_mask_position(mask_path: str) -> MaskPos:
    mask_img_alpha = np.array(pil_img.open(mask_path).convert("RGBA"))[:, :, 3]
    height, width = mask_img_alpha.shape

    top = None
    bottom = None
    left = None
    right = None

    for i_row in range(height):
        if (mask_img_alpha[i_row, :] > 0).any():
            top = i_row / height
            break

    for i_row in reversed(range(height)):
        if (mask_img_alpha[i_row, :] > 0).any():
            bottom = (i_row + 1) / height
            break

    for i_col in range(width):
        if (mask_img_alpha[:, i_col] > 0).any():
            left = i_col / width
            break

    for i_col in reversed(range(width)):
        if (mask_img_alpha[:, i_col] > 0).any():
            right = (i_col + 1) / width
            break

    assert top is not None, "No top found"
    assert left is not None, "No left found"

    return MaskPos(left, top, right - left, bottom - top)


def shuffled(l: list) -> list:
    l = copy.deepcopy(l)
    random.shuffle(l)
    return l


def process_ds_folder(
    folder: str,
    hotels_50k: bool,
    min_sample_limit: int,
    max_sample_limit: Union[int, float],
) -> DsResults:
    all_imgs = utils.get_train_img_paths(
        os.path.join(folder, "train_images") if not hotels_50k else folder
    )

    labels = utils.paths_to_labels(all_imgs, hotels_50k=hotels_50k)
    label_counts = defaultdict(int)
    for l in labels:
        label_counts[l] += 1

    correct_amount_idxs = [
        i
        for i, l in enumerate(labels)
        if min_sample_limit <= label_counts[l] <= max_sample_limit
    ]
    sampled_idxs_per_label = defaultdict(list)

    for i, l in shuffled(
        [(i, l) for i, l in enumerate(labels) if label_counts[l] > max_sample_limit]
    ):
        if len(sampled_idxs_per_label[l]) < max_sample_limit:
            sampled_idxs_per_label[l].append(i)

    sampled_idxs = []
    for idxs in sampled_idxs_per_label.values():
        assert len(idxs) <= max_sample_limit
        sampled_idxs.extend(idxs)

    remaining_idxs = correct_amount_idxs + sampled_idxs
    all_imgs = utils.list_index(all_imgs, remaining_idxs)
    labels = utils.list_index(labels, remaining_idxs)

    label_encoder, label_decoder = utils.make_label_map(labels)
    cls_counts = utils.class_counts(labels)

    single_class_entries = [cls_ for (cls_, count) in cls_counts.items() if count == 1]

    for cls_ in single_class_entries:
        idx = labels.index(cls_)

        all_imgs.append(all_imgs[idx])
        labels.append(labels[idx])

    train_idxs, val_idxs = train_test_split(
        list(range(len(all_imgs))),
        test_size=0.1,
        stratify=labels,
        random_state=42,
    )

    assert isinstance(train_idxs, list)
    assert isinstance(val_idxs, list)

    return DsResults(
        images=all_imgs,
        labels=labels,
        label_encoder=label_encoder,
        label_decoder=label_decoder,
        train_idxs=train_idxs,
        val_idxs=val_idxs,
    )


@click.command()
@click.option("--data-folder", "-d", type=click.Path(exists=True))
@click.option("--hotels-50k", is_flag=True, show_default=True, type=bool, default=False)
@click.option("--mask-folder", "-m", default=None, type=click.Path(exists=True))
@click.option("--output-file", "-o", type=click.Path())
@click.option(
    "--min-sample-limit",
    type=int,
    default=1,
    help="Minimal amount of samples required for a class to be included",
)
@click.option(
    "--max-sample-limit",
    type=int,
    default=None,
    help="Maximal amount of samples of a class for it to be included",
)
def main(
    data_folder,
    hotels_50k: bool,
    mask_folder: Optional[str],
    output_file,
    min_sample_limit: int,
    max_sample_limit: Optional[int],
):
    if max_sample_limit is None:
        max_sample_limit = math.inf

    all_mask_imgs = utils.get_mask_img_paths(
        os.path.join(data_folder, "train_masks") if mask_folder is None else mask_folder
    )

    ds_results = process_ds_folder(
        data_folder, hotels_50k, min_sample_limit, max_sample_limit
    )

    mask_positions = Parallel(n_jobs=-1)(
        delayed(extr_mask_position)(m) for m in tqdm(all_mask_imgs)
    )

    metadata = TrainMetadata(
        ds_results.label_encoder,
        ds_results.label_decoder,
        ds_results.images,
        ds_results.labels,
        ds_results.train_idxs,
        ds_results.val_idxs,
        mask_positions,
    )

    print(
        f"Created metadata with: {len(metadata.images)} samples, with {len(ds_results.label_decoder)} unique classes"
    )
    metadata.to_yaml(output_file)


if __name__ == "__main__":
    main()
