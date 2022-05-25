from typing import Any, Dict, Optional
import pytorch_lightning as pl
import timm
import timm.optim
import torch.optim
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from library.metric import mean_average_precision


class TimmModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        extra_model_params: Optional[Dict[str, Any]] = None,
        pretrained=True,
    ) -> None:
        super().__init__()

        self._model_name = model_name
        self._n_classes = n_classes
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._extra_model_params = (
            extra_model_params if extra_model_params is not None else dict()
        )

        self.model = timm.create_model(
            self._model_name,
            pretrained=pretrained,
            num_classes=self._n_classes,
            **self._extra_model_params,
        )

        self.loss_fn = F.cross_entropy

        self.save_hyperparameters()

    def get_transform(self):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)

        return transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, *_) -> Dict:

        inputs, labels = batch

        predictions = self.forward(inputs)

        loss = self.loss_fn(predictions, labels)
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *_) -> Dict:
        inputs, labels = batch
        predictions = self.forward(inputs)
        return {"predictions": predictions, "ground_truth": labels}

    def validation_epoch_end(self, outputs) -> None:

        predictions = []
        labels = []
        for output in outputs:
            predictions.append(output["predictions"])
            labels.append(output["ground_truth"])

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

        val_loss = self.loss_fn(predictions, labels)
        val_acc = accuracy_score(
            labels.cpu().numpy(), predictions.argmax(dim=1).cpu().numpy()
        )
        val_map5 = mean_average_precision(predictions.detach().cpu().numpy(), labels)

        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        self.log("val_map5", val_map5)

        top_5 = torch.topk(predictions, 5).indices

        with open(".cache/val_preds", "w") as f:
            for i in range(top_5.shape[0]):
                f.write(f"{labels[i]}\t")
                for j in range(top_5.shape[1]):
                    f.write(f"{top_5[i, j]} ")
                f.write("\n")

    def configure_optimizers(self):

        optimizer = timm.optim.create_optimizer_v2(
            self.model,
            opt=self._optimizer,
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return [optimizer], [scheduler]


# --------------------------------------------------------- for rotation model -------------------------------------------------


class TimmModule_rotation(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        extra_model_params: Optional[Dict[str, Any]] = None,
        pretrained=True,
    ) -> None:
        super().__init__()

        self._model_name = model_name
        self._n_classes = n_classes
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._extra_model_params = (
            extra_model_params if extra_model_params is not None else dict()
        )

        self.model = timm.create_model(
            self._model_name,
            pretrained=pretrained,
            num_classes=self._n_classes,
            **self._extra_model_params,
        )

        self.loss_fn = F.cross_entropy

        self.save_hyperparameters()

    def get_transform(self):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)

        return transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, *_) -> Dict:

        inputs, labels = batch

        predictions = self.forward(inputs)

        loss = self.loss_fn(predictions, labels)
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *_) -> Dict:
        inputs, labels = batch
        predictions = self.forward(inputs)
        return {"predictions": predictions, "ground_truth": labels}

    def validation_epoch_end(self, outputs) -> None:

        predictions = []
        labels = []
        for output in outputs:
            predictions.append(output["predictions"])
            labels.append(output["ground_truth"])

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

        val_loss = self.loss_fn(predictions, labels)
        val_acc = accuracy_score(
            labels.cpu().numpy(), predictions.argmax(dim=1).cpu().numpy()
        )
        # val_map5 = mean_average_precision(predictions.detach().cpu().numpy(), labels)

        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        # self.log("val_map5", val_map5)

        # top_5 = torch.topk(predictions, 5).indices

        # with open(".cache/val_preds", "w") as f:
        #     for i in range(top_5.shape[0]):
        #         f.write(f"{labels[i]}\t")
        #         for j in range(top_5.shape[1]):
        #             f.write(f"{top_5[i, j]} ")
        #         f.write("\n")

    def configure_optimizers(self):

        optimizer = timm.optim.create_optimizer_v2(
            self.model,
            opt=self._optimizer,
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        return [optimizer], [scheduler]
