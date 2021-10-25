from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tmet
from typing import Any, Tuple, Type
from . import experiment, model
from .data import Data
from .data.cinc2017 import Cinc2017

@dataclass
class Metrics(experiment.Metrics):
    accuracy: float
    accuracy_perclass: np.ndarray
    f1: float
    f1_nao: float
    f1_perclass: np.ndarray
    auroc: float
    auroc_nao: float
    auroc_perclass: np.ndarray
    confusion: np.ndarray


@dataclass
class Params(experiment.Params):
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
    def __init__(self) -> None:
        super().__init__()
        n_cls = len(Cinc2017.CATS)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = tmet.Accuracy(compute_on_step=False, num_classes=n_cls)
        self.accuracy_perclass = tmet.Accuracy(compute_on_step=False, num_classes=n_cls, average=None)
        self.f1 = tmet.F1(compute_on_step=False, num_classes=n_cls, average=None)
        self.auroc = tmet.AUROC(compute_on_step=False, num_classes=n_cls, average=None)
        self.confusion = tmet.ConfusionMatrix(compute_on_step=False, num_classes=n_cls)

    def _step(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, int]:
        x, y = batch
        z = model(x)
        loss = self.loss(z, y)
        with torch.no_grad():
            zmax = torch.argmax(z, dim=1)
            self.accuracy(zmax, y)
            self.accuracy_perclass(zmax, y)
            self.f1(zmax, y)
            self.auroc(torch.softmax(z, dim=1), y)
            self.confusion(zmax, y)
        return loss, y.shape[0]

    def compute(self) -> Metrics:
        f1 = self.f1.compute().cpu().numpy()
        auroc = self.auroc.compute().cpu().numpy()
        return Metrics(
            loss = self.compute_loss(),
            accuracy = self.accuracy.compute().item(),
            accuracy_perclass = self.accuracy_perclass.compute().cpu().numpy(),
            f1 = f1.mean(),
            f1_nao = f1[:-1].mean(),
            f1_perclass = f1,
            auroc = auroc.mean(),
            auroc_nao = auroc[:-1].mean(),
            auroc_perclass = auroc,
            confusion = self.confusion.compute().cpu().numpy())


class Experiment(experiment.Experiment):
    def default_params(self) -> Params:
        return Params()

    def metrics_class(self) -> Type[Metrics]:
        return Metrics

    def default_name(self) -> str:
        return 'classify_cinc2017'

    def data(self, base_path: str, params: Params, folds: int, rseed: int) -> Data:
        data = Cinc2017(base_path=base_path, n_folds=folds, split_seed=rseed)
        return data(params.batch_size, params.trim_prob, params.trim_min)

    def model(self, params: Params) -> nn.Module:
        return model.Classify(
            features = Cinc2017.N_FEATURES,
            classes = len(Cinc2017.CATS),
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

    def step(self) -> Step:
        return Step()


if __name__=='__main__':
    Experiment().main()
