from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import yaml
import pathlib
from copy import deepcopy
from nils_utils.experiment_tools import load_from_yaml


@dataclass
class MaskPos:
    """
    contains mask pos in percents
    """

    left: float
    top: float
    width: float
    height: float

    def _values(self) -> List[float]:
        return [self.left, self.top, self.width, self.height]

    def to_indices(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        return (
            int(self.left * img_width),
            int(self.top * img_height),
            int(self.width * img_width),
            int(self.height * img_height),
        )


@dataclass
class TrainMetadata:
    label_encoder: Dict[str, int]
    label_decoder: Dict[int, str]
    images: List[str]
    txt_labels: List[str]
    train_idxs: List[str]
    val_idxs: List[str]
    mask_positions: List[MaskPos]

    def to_yaml(self, file_path) -> None:
        metadata = {
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "images": self.images,
            "txt_labels": self.txt_labels,
            "train_idxs": self.train_idxs,
            "val_idxs": self.val_idxs,
            "mask_positions": [mp._values() for mp in self.mask_positions],
        }

        with open(file_path, "w") as f:
            yaml.dump(metadata, f)

    @staticmethod
    def from_yaml(yaml_file: str) -> "TrainMetadata":

        with open(yaml_file, "r") as f:

            metadata = yaml.load(f, Loader=yaml.CLoader)

        return TrainMetadata(
            label_encoder=metadata["label_encoder"],
            label_decoder=metadata["label_decoder"],
            images=metadata["images"],
            txt_labels=metadata["txt_labels"],
            train_idxs=metadata["train_idxs"],
            val_idxs=metadata["val_idxs"],
            mask_positions=[MaskPos(*vs) for vs in metadata["mask_positions"]],
        )


@load_from_yaml
@dataclass
class ExpConfig:
    model_type: str
    experiment_name: str = "Default"
    num_epochs: int = 100
    batch_size: int = 32
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    augmentation_preset: str = "default"  # see library.data.augmentations
    val_augmentation_preset: str = "default"
    gradient_accumulation: int = 1
    extra_model_params: Optional[Dict] = None
    pretrained_timm_model: Optional[str] = None
    use_arcface_loss: bool = False


@load_from_yaml
@dataclass
class InferenceConfig:
    model_path: str
    val_augmentation_preset: str = "default"
    batch_size: int = 16
    embedding_based: bool = False
    rot_model_ckpt: Optional[str] = None
    embedding_rot_cor: bool = False


@load_from_yaml
@dataclass
class MetaExpConfig:
    experiment_name: str = "MetaLearn"
    n_folds: int = 5
