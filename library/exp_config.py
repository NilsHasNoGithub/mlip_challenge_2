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
    train_imgs: List[str]
    val_imgs: List[str]
    mask_positions: List[MaskPos]

    def to_yaml(self, file_path) -> None:
        metadata = {
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "train_imgs": self.train_imgs,
            "val_imgs": self.val_imgs,
            "mask_positions": [mp._values() for mp in self.mask_positions],
        }

        with open(file_path, "w") as f:
            yaml.dump(metadata, f)

    @staticmethod
    def from_yaml(yaml_file: str) -> "TrainMetadata":

        with open(yaml_file, "r") as f:

            metadata = yaml.load(f, Loader=yaml.FullLoader)

        return TrainMetadata(
            label_encoder=metadata["label_encoder"],
            label_decoder=metadata["label_decoder"],
            train_imgs=metadata["train_imgs"],
            val_imgs=metadata["val_imgs"],
            mask_positions=[MaskPos(*vs) for vs in metadata["mask_positions"]],
        )


@load_from_yaml
@dataclass
class ExpConfig:
    model_type: str
    experiment_name: str = "Default"
    num_epochs: int = 100
    batch_size: int = 32
    optimizer: str = "sgf"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    augmentation_preset: str = "default"  # see library.data.augmentations
    val_augmentation_preset: str = "default"
    gradient_accumulation: int = 1
    extra_model_params: Optional[Dict] = None

    # @staticmethod
    # def _defaults() -> "ExpConfig":
    #     return ExpConfig(None)

    # @staticmethod
    # def from_dict(config: dict) -> "ExpConfig":
    #     default = ExpConfig._defaults()
    #     return ExpConfig(
    #         model_type=config["model_type"],
    #         experiment_name=config.get("experiment_name", default.experiment_name),
    #         num_epochs=config.get("num_epochs", default.num_epochs),
    #         batch_size=config.get("batch_size", default.batch_size),
    #         optimizer=config.get("optimizer", default.optimizer),
    #         learning_rate=config.get("learning_rate", default.learning_rate),
    #         weight_decay=config.get("weight_decay", default.weight_decay),
    #         augmentation_preset=config.get(
    #             "augmentation_preset", default.augmentation_preset
    #         ),
    #         gradient_accumulation=config.get(
    #             "gradient_accumulation", default.gradient_accumulation
    #         )
    #     )

    # @staticmethod
    # def from_yaml_file(yaml_file: Union[str, pathlib.Path]) -> "ExpConfig":
    #     with open(yaml_file) as f:
    #         return ExpConfig.from_dict(yaml.load(f, Loader=yaml.FullLoader))

    # @staticmethod
    # def load_multi_conf(yaml_file: Union[str, pathlib.Path]) -> List["ExpConfig"]:
    #     """
    #     The format:
    #     Object `base`: default experiment configuration, containing all hyperparameters
    #     List `deltas`: List of changes in hyperparameters compared to base, all deltas result in a new configuration

    #     ## params
    #     - `yaml_file`: file containing the configuration

    #     ## returns
    #     A list of experiment configurations, the first of which is the base config
    #     """

    #     with open(yaml_file, "r") as f:
    #         multi_conf = yaml.load(f, Loader=yaml.FullLoader)

    #     base_dict: Dict = multi_conf["base"]
    #     result = [ExpConfig.from_dict(base_dict)]

    #     if "deltas" not in multi_conf.keys():
    #         return result

    #     deltas: List[Dict] = multi_conf["deltas"]
    #     for delta in deltas:
    #         delta_dict = deepcopy(base_dict)
    #         delta_dict.update(delta)
    #         result.append(ExpConfig.from_dict(delta_dict))

    #     return result
