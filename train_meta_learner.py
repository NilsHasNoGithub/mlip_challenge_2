import os
from typing import Optional
import click
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import numpy as np

from library.config import MetaExpConfig
from os.path import join as pjoin
import h2o

from h2o.automl import H2OAutoML


@click.command()
@click.option(
    "--data-folder",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="folder containing ['train_feats.npy', 'train_labels.npy'], and optionally: ['val_feats.npy', 'val_labels.npy']",
)
@click.option(
    "--output-folder",
    "-o",
    required=True,
    type=click.Path(),
    help="Path to save outputs",
)
@click.option(
    "--train-config",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Train experiment configuration file",
)
@click.option(
    "--num-threads", "-j", default=-1, type=int, help="number of threads for h2o to use"
)
@click.option("--runtime-mins", "-r", default=5 * 24 * 60, type=int)  # default 5 days
@click.option(
    "--max-mem-size", default=None, type=str, help="Maximum h2o cluster memory"
)
@click.option(
    "--min-mem-size", default=None, type=str, help="Minimum h2o cluster memory"
)
def main(
    data_folder: str,
    train_config: str,
    output_folder: str,
    runtime_mins: int,
    num_threads: int,
    max_mem_size: Optional[str],
    min_mem_size: Optional[str],
):
    runtime_secs = runtime_mins * 60

    h2o.init(nthreads=num_threads, min_mem_size=min_mem_size, max_mem_size=max_mem_size)
    os.makedirs(output_folder, exist_ok=True)

    train_config: MetaExpConfig = MetaExpConfig.from_yaml_file(train_config)

    train_feats: np.ndarray = np.load(pjoin(data_folder, "train_feats.npy"))
    train_labels: np.ndarray = np.load(pjoin(data_folder, "train_labels.npy"))

    val_feats_path = pjoin(data_folder, "val_feats.npy")
    val_labels_path = pjoin(data_folder, "val_labels.npy")

    n_feats = train_feats.shape[1]

    h2o_train_dict = {f"feat{i}": train_feats[:, i] for i in tqdm(range(n_feats))}

    feat_colnames = [f for f in h2o_train_dict.keys()]

    label_colname = "label"
    h2o_train_dict[label_colname] = list(train_labels)

    pd_dataframe = pd.DataFrame.from_dict(h2o_train_dict)
    train_data = h2o.H2OFrame(pd_dataframe)

    if os.path.exists(val_feats_path) and os.path.exists(val_labels_path):
        val_feats: np.ndarray = np.load(val_feats_path)
        val_labels: np.ndarray = np.load(val_labels_path)

        h2o_val_dict = {f"feat{i}": list(val_feats[:, i]) for i in tqdm(range(n_feats))}
        h2o_val_dict[label_colname] = list(val_labels)

        val_data = h2o.H2OFrame(h2o_val_dict)
        val_data[label_colname] = val_data[label_colname].asfactor()
    else:
        val_data = None

    train_data[label_colname] = train_data[label_colname].asfactor()

    aml = H2OAutoML(max_runtime_secs=runtime_secs)
    aml.train(x=feat_colnames, y=label_colname, training_frame=train_data)

    best_model = aml.get_best_model()
    best_model_path = h2o.save_model(best_model, output_folder)

    full_model_path = h2o.save_model(aml, output_folder)

    # print paths
    print(f"Best model path: {best_model_path}")
    print(f"Full model path: {full_model_path}")

    # val_feats: np.ndarray = np.load(pjoin(data_folder, "val_feats.npy"))
    # val_labels: np.ndarray = np.load(pjoin(data_folder, "val_labels.npy"))


if __name__ == "__main__":
    main()
