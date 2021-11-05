from abc import ABC, abstractmethod
import torch.utils.data as tud
from typing import Iterator, Tuple


DataLoaders = Iterator[Tuple[tud.DataLoader, tud.DataLoader]]


class Data(ABC):
    @property
    @abstractmethod
    def n_features(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> DataLoaders:
        raise NotImplementedError()
