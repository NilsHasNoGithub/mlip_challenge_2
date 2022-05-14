from genericpath import exists
import os
import pathlib
from typing import List
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch
from library.data.dataset import HotelLightningModule
from library.config import ExpConfig, TrainMetadata
import click
from library.models.timm_model import TimmModule
from library.data import augmentations
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import multiprocessing
import random
from icecream import ic
import warnings


# def random_str(l=10) ->


def unique_str(l=10) -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=l))


@click.command()
@click.option(
    "--train-metadata",
    "-t",
    type=click.Path(exists=True),
    help="Path to train metadata file",
)
@click.option(
    "--experiment-conf",
    "-e",
    type=click.Path(exists=True),
    help="Path to experiment configuration",
)
@click.option("--num-dl-workers", type=int, default=multiprocessing.cpu_count() // 2)
@click.option("--multi-gpu/--no-multi-gpu", type=bool, default=False)
@click.option(
    "--gpu",
    multiple=True,
    type=int,
    default=[i for i in range(torch.cuda.device_count())],
    help="GPUs to use",
)
def main(
    train_metadata: str,
    experiment_conf: str,
    num_dl_workers: int,
    multi_gpu: bool,
    gpu: List[int],
):
    warnings.simplefilter(action="ignore", category=FutureWarning)
    train_metadata = TrainMetadata.from_yaml(train_metadata)
    exp_configs: List[ExpConfig] = ExpConfig.from_multi_conf_yaml_file(experiment_conf)

    n_classes = len(train_metadata.label_encoder)

    gpus = gpu

    for exp_config in exp_configs:

        print(f"Running experiment:\n{exp_config}")

        logger = MLFlowLogger(experiment_name=exp_config.experiment_name)
        unique_str = logger.run_id[:10]

        for k, v in vars(exp_config).items():
            logger.experiment.log_param(logger.run_id, f"{k}__", v)

        exts_to_bup = [".py", ".zsh", ".bash", ".sh", ".yml", ".yaml"]
        files_to_bup = []

        for ext in exts_to_bup:
            files_to_bup.extend(pathlib.Path.cwd().glob(f"**/*{ext}"))

        for file in files_to_bup:
            file_rel = file.relative_to(pathlib.Path.cwd())

            # prevent recursion
            if str(file_rel).startswith("mlruns/"):
                continue

            logger.experiment.log_artifact(
                logger.run_id, str(file), str(pathlib.Path("code") / file_rel.parent)
            )

        model = TimmModule(
            exp_config.model_type,
            n_classes=n_classes,
            optimizer=exp_config.optimizer,
            learning_rate=exp_config.learning_rate,
            weight_decay=exp_config.weight_decay,
            extra_model_params=exp_config.extra_model_params,
        )

        transform = ic(model.get_transform())

        data_module = HotelLightningModule(
            train_metadata,
            exp_config,
            num_dl_workers=num_dl_workers,
            augmentation_pipeline=augmentations.PRESETS[exp_config.augmentation_preset],
            val_augmentation_pipeline=augmentations.VAL_PRESETS[
                exp_config.val_augmentation_preset
            ],
            transform=transform,
        )

        model_dir = os.path.join("models", unique_str)
        os.makedirs(model_dir, exist_ok=True)
        logger.experiment.log_param(logger.run_id, "model_dir", model_dir)
        pattern = "epoch-{epoch:04d}_val-map5-{val_map5:.4f}"
        ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
        checkpointers = []
        for metr in ["val_map5"]:
            checkpointer = ModelCheckpoint(
                dirpath=model_dir,
                monitor=metr,
                filename=pattern + ".best",
                save_last=True,
                auto_insert_metric_name=False,
                save_top_k=3,
                mode="max",
            )
            checkpointers.append(checkpointer)

        trainer = pl.Trainer(
            max_epochs=exp_config.num_epochs,
            accelerator="gpu",
            logger=logger,
            callbacks=[*checkpointers, LearningRateMonitor()],
            devices=gpus,
            # strategy="ddp_sharded",
            # strategy=DeepSpeedStrategy(
            #     stage=3,
            #     # offload_parameters=True,
            #     offload_optimizer=True,
            #     cpu_checkpointing=True,
            # ),
            # strategy="ddp_sharded",
            # strategy="ddp_fully_sharded" if multi_gpu else None,
            # auto_select_gpus=True,
            accumulate_grad_batches=exp_config.gradient_accumulation,
        )
        # trainer = pl.Trainer(logger=logger, gpus=[2])

        trainer.fit(model, data_module)

        del model
        del data_module
        del trainer


if __name__ == "__main__":
    main()
