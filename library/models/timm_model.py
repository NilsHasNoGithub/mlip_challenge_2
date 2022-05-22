import os
from turtle import forward
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pytorch_lightning as pl
import timm
import timm.optim
import torch.optim
import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from library.layers.arc_face_loss import AngularPenaltySMLoss

from library.metric import mean_average_precision, mean_average_precision_topk
from icecream import ic


class Head(nn.Module):
    def __init__(self, num_features, num_classes, use_arcface, drop_rate) -> None:
        super().__init__()

        self._dropout = nn.Dropout(drop_rate)
        self._use_arcface = use_arcface

        self._layer = (
            AngularPenaltySMLoss(num_features, num_classes, m=0.3, s=30.0)
            if use_arcface
            else nn.Linear(num_features, num_classes)
        )

    def forward(
        self, x: torch.Tensor, labels=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self._dropout(x)

        if self._use_arcface:
            assert labels is not None
            x, loss = self._layer(x, labels)
            return x, loss

        x = self._layer(x)
        return x, None


class TimmModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        extra_model_params: Optional[Dict[str, Any]] = None,
        pretrained_timm_model: Optional[str] = None,
        pretrained=True,
        head_drop_rate: float = 0.5,
        label_smoothing: float = 0.1,
        for_inference: bool = False,
        use_arcface_loss: bool = False,
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
        self._for_inference = for_inference
        self._use_arcface_loss = use_arcface_loss

        self._running_embeddings: List[Tuple[torch.Tensor, int]] = []
        self._latest_embeddings: Optional[List[Tuple[torch.Tensor, int]]] = None

        if pretrained_timm_model is not None:
            self.model = torch.load(pretrained_timm_model)
            print("loading pretrained model")
        else:
            self.model = timm.create_model(
                self._model_name,
                pretrained=pretrained,
                num_classes=0,
                **self._extra_model_params
            )

        self.head = Head(
            self.model.num_features, self._n_classes, use_arcface_loss, head_drop_rate
        )

        if for_inference:
            self.loss_fn = nn.CrossEntropyLoss()  # for old pytorch version on kaggle
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.save_hyperparameters()

    def get_transform(self):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)

        return transform

    def forward_features(self, x: torch.Tensor):
        return self.model(x)

    def forward(
        self, x: torch.Tensor, labels=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.model(x)
        x = self.head(x, labels=labels)
        return x

    def create_embeddings(self, batch_iter: Iterable, discard_old=True, use_tqdm=True):
        """ """
        if discard_old:
            self._latest_embeddings = []

        for inputs, lbls in tqdm(batch_iter, disable=not use_tqdm):
            embeddings = self.forward_features(inputs.to(self.device)).detach().cpu()

            for i in range(embeddings.shape[0]):
                embed = embeddings[i, :]
                lbl = lbls[i].item()

                self._latest_embeddings.append((embed, lbl))

    def embedding_db(self) -> torch.Tensor:
        if self._latest_embeddings is None:
            raise Exception("Embeddings not set, call `create_embeddings` first")
        return torch.stack([t for t, _ in self._latest_embeddings]).detach().cpu()

    def label_db(self) -> torch.Tensor:
        if self._latest_embeddings is None:
            raise Exception("Embeddings not set, call `create_embeddings` first")
        return torch.tensor([l for _, l in self._latest_embeddings]).detach().cpu()

    def predict_based_on_embedding(
        self, inputs: torch.Tensor, k=5, embedding_db=None, lbl_db=None
    ) -> List[List[int]]:
        """

        ## parameters
        - `inputs`: batch of images
        - `k`: top predictions to return
        """

        embedding_db = self.embedding_db() if embedding_db is None else embedding_db
        lbl_db = self.label_db() if lbl_db is None else lbl_db

        result = []

        embeddings = self.forward_features(inputs.to(self.device)).cpu().detach()

        for i in range(embeddings.shape[0]):
            top_k = self.top_k_based_on_embedding(
                embeddings[i, :], embedding_db=embedding_db, lbl_db=lbl_db, k=k
            )

            result.append(top_k)

        return result

    def top_k_based_on_embedding(
        self, embedding: torch.Tensor, k=5, embedding_db=None, lbl_db=None
    ) -> List[int]:
        embedding_db = self.embedding_db() if embedding_db is None else embedding_db
        lbl_db = self.label_db() if lbl_db is None else lbl_db

        cossim = nn.CosineSimilarity()
        embedding = embedding.detach().cpu()

        if len(embedding.shape) < 2:
            embedding = embedding.view(1, -1)

        similarities = cossim(embedding_db, embedding)

        sorted_args = torch.argsort(similarities, descending=True)

        top_k = []
        i = 0

        while len(top_k) < k:
            pred_lbl = lbl_db[sorted_args[i]]

            if isinstance(pred_lbl, torch.Tensor):
                pred_lbl = pred_lbl.item()

            if pred_lbl not in top_k:
                top_k.append(pred_lbl)

            i += 1

        return top_k

    def training_step(self, batch, *_) -> Dict:

        inputs, labels = batch

        features = self.forward_features(inputs)
        predictions, loss = self.head(features, labels=labels)

        if not self._use_arcface_loss:
            assert loss is None
            loss = self.loss_fn(predictions, labels)
        else:
            assert loss is not None

        self.log("loss", loss)

        for i in range(features.shape[0]):
            i_feats = features[i, :].detach().cpu()
            i_lbl = labels[i].item()

            self._running_embeddings.append((i_feats, i_lbl))

        return {"loss": loss}

    def training_epoch_end(self, _outputs) -> None:
        self._latest_embeddings = self._running_embeddings
        self._running_embeddings = []

    def validation_step(self, batch, *_) -> Dict:
        inputs, labels = batch
        embeddings = self.forward_features(inputs)
        predictions, loss = self.head(embeddings, labels=labels)

        return {
            "predictions": predictions,
            "embeddings": embeddings,
            "ground_truth": labels,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs) -> None:

        embeddings = []
        predictions = []
        labels = []
        val_losses = []
        for output in outputs:
            embeds = output["embeddings"]
            for i in range(embeds.shape[0]):
                embeddings.append(embeds[i, :])

            predictions.append(output["predictions"])
            labels.append(output["ground_truth"])
            val_losses.append(output["val_loss"])

        if self._use_arcface_loss:
            val_loss = np.mean([v.item() for v in val_losses])
        else:
            val_loss = self.loss_fn(predictions, labels)

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

        val_acc = accuracy_score(
            labels.cpu().numpy(), predictions.argmax(dim=1).cpu().numpy()
        )
        val_map5 = mean_average_precision(predictions.detach().cpu().numpy(), labels)

        if self._latest_embeddings is not None and len(self._latest_embeddings) > 0:
            embedding_db = self.embedding_db()
            lbl_db = self.label_db()

            top_k_preds = []
            for embed in embeddings:
                top_k_preds.append(
                    self.top_k_based_on_embedding(
                        embed, embedding_db=embedding_db, lbl_db=lbl_db
                    )
                )

            val_map5_embed = mean_average_precision_topk(top_k_preds, labels.tolist())
            self.log("val_map5_embed", val_map5_embed)

        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        self.log("val_map5", val_map5)

    def configure_optimizers(self):

        optimizer = timm.optim.create_optimizer_v2(
            self.parameters(),
            opt=self._optimizer,
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 10, eta_min=self._learning_rate * 1e-2
        )
        return [optimizer], [scheduler]
