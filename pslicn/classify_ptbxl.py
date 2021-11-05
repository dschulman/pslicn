from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tmet
from typing import Any, List, Tuple, Type
from . import experiment, model
from .data import Data
from .data.ptbxl import Ptbxl, Task


@dataclass
class Metrics(experiment.Metrics):
    accuracy: float
    accuracy_perclass: np.ndarray
    f1: float
    f1_perclass: np.ndarray
    auroc: float
    auroc_perclass: np.ndarray
    confusion: np.ndarray

    def summary(self) -> str:
        return f'Loss={self.loss:.3f} Acc={self.accuracy:.3f} F1={self.f1:.3f} AUC={self.auroc:.3f}'


@dataclass
class Params(experiment.Params):
    task: Task = Task.All
    high_res: bool = False
    inproj_size: int = 7
    inproj_stride: int = 4
    inproj_norm: bool = False
    hidden: int = 64
    kernel_size: int = 5
    stride: int = 2
    layers: int = 2
    depth_variant: bool = True
    stride_on: model.StrideOn = model.StrideOn.All
    outproj_size: int = 64
    dropout: float = 0.1
    leak: float = 0.0
    layer_norm: bool = False
    batch_size: int = 32
    trim_prob: float = 0.9
    trim_min: float = 0.5
    lr: float = 0.001


class Step(experiment.Step):
    def __init__(self, cats: List[str]) -> None:
        super().__init__()
        n_cls = len(cats)
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = tmet.Accuracy(
            compute_on_step=False,
            num_classes=n_cls,
            average=None,
            multiclass=False,
            threshold=0.0)
        self.f1 = tmet.F1(
            compute_on_step=False,
            num_classes=n_cls,
            average=None,
            multiclass=False,
            threshold=0.0)
        self.auroc = tmet.AUROC(
            compute_on_step=False,
            num_classes=n_cls,
            average=None,
            pos_label=1)
        self.confusion = tmet.ConfusionMatrix(
            compute_on_step=False,
            num_classes=n_cls,
            multilabel=True,
            threshold=0.0)

    def _step(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, int]:
        x, y = batch
        z = model(x)
        loss = self.loss(z, y.float())
        with torch.no_grad():
            self.accuracy(z, y)
            self.f1(z, y)
            self.auroc(z, y)
            self.confusion(z, y)
        return loss, y.shape[0]

    def compute(self) -> Metrics:
        accuracy = self.accuracy.compute().cpu().numpy()
        f1 = self.f1.compute().cpu().numpy()
        auroc = self.auroc.compute().cpu().numpy()
        return Metrics(
            loss = self.compute_loss(),
            accuracy = accuracy.mean(),
            accuracy_perclass = accuracy,
            f1 = f1.mean(),
            f1_perclass = f1,
            auroc = auroc.mean(),
            auroc_perclass = auroc,
            confusion = self.confusion.compute().cpu().numpy())


class Experiment(experiment.Experiment):
    def default_params(self) -> Params:
        return Params()

    def metrics_class(self) -> Type[Metrics]:
        return Metrics

    def default_name(self) -> str:
        return 'classify_ptbxl'

    def data(self, base_path: str, params: Params, folds: int, rseed: int) -> Data:
        return Ptbxl(
            task = params.task,
            base_path = base_path,
            high_res = params.high_res,
            batch_size = params.batch_size,
            trim_prob = params.trim_prob,
            trim_min = params.trim_min)

    def model(self, params: Params, data: Ptbxl) -> nn.Module:
        return model.Classify(
            features = Ptbxl.N_FEATURES,
            classes = len(data.cats),
            inproj_size = params.inproj_size,
            inproj_stride = params.inproj_stride,
            inproj_norm = params.inproj_norm,
            hidden = params.hidden,
            kernel_size = params.kernel_size,
            stride = params.stride,
            stride_on = params.stride_on,
            layers = params.layers,
            depth_variant = params.depth_variant,
            outproj_size = params.outproj_size,
            dropout = params.dropout,
            leak = params.leak,
            layer_norm = params.layer_norm)

    def optimizer(self, model: nn.Module, params: Params) -> optim.Optimizer:
        return optim.Adam(
            params = model.parameters(),
            lr = params.lr)

    def step(self, data: Ptbxl) -> Step:
        return Step(data.cats)


if __name__=='__main__':
    Experiment().main()
