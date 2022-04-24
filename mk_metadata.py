from genericpath import exists
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


@click.command()
@click.option("--data-folder", "-d", type=click.Path(exists=True))
@click.option("--output-file", "-o", type=click.Path())
def main(data_folder, output_file):
    all_train_imgs = utils.get_train_img_paths(
        os.path.join(data_folder, "train_images")
    )
    all_mask_imgs = utils.get_mask_img_paths(os.path.join(data_folder, "train_masks"))

    labels = utils.paths_to_labels(all_train_imgs)

    label_encoder, label_decoder = utils.make_label_map(labels)
    cls_counts = utils.class_counts(labels)

    single_class_entries = [cls_ for (cls_, count) in cls_counts.items() if count == 1]

    for cls_ in single_class_entries:
        idx = labels.index(cls_)

        all_train_imgs.append(all_train_imgs[idx])
        labels.append(labels[idx])

    train_idxs, val_idxs = train_test_split(
        list(range(len(all_train_imgs))), test_size=0.1, stratify=labels, random_state=42
    )

    assert isinstance(train_idxs, list)
    assert isinstance(val_idxs, list)

    mask_positions = Parallel(n_jobs=-1)(
        delayed(extr_mask_position)(m) for m in tqdm(all_mask_imgs)
    )

    metadata = TrainMetadata(
        label_encoder, label_decoder, all_train_imgs, train_idxs, val_idxs, mask_positions
    )

    metadata.to_yaml(output_file)


if __name__ == "__main__":
    main()
