import numpy as np
import os
import pandas as pd
import scipy.io as spio
import sklearn.model_selection as skms
import time
import torch
import torch.utils.data as tud
from typing import List, Tuple
from . import Data, DataLoaders, utils
from .utils import _Dataset, _Augment


class Cinc2017(Data):
    N_FEATURES = 1
    CATS = ['N', 'A', 'O', '~']
    URL = 'https://www.physionet.org/files/challenge-2017/1.0.0/training2017.zip?download'
    PATH = 'cinc2017'

    def __init__(
            self,
            base_path: str = 'data',
            n_folds: int = 5,
            train_pct: float = 0.7,
            split_seed: int = 1234,
            batch_size: int = 32,
            trim_prob: float = 0.9,
            trim_min: float = 0.5,
    ) -> None:
        self.path = os.path.join(base_path, self.PATH)
        self.n_folds = n_folds
        self.train_pct = train_pct
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.trim_prob = trim_prob
        self.trim_min = trim_min
        self.n_classes = len(self.CATS)
        utils.download_and_unzip(self.URL, 'training2017', self.path)
        self.datasets = self._setup()
    @property
    def n_features(self):
        return 1

    def _setup(self) -> List[Tuple[_Dataset, _Dataset]]:
        ref_path = os.path.join(self.path, 'REFERENCE.csv')
        ref = pd.read_csv(ref_path, names=['id', 'label'])
        xs = np.empty(ref.shape[0], dtype=np.object)
        start_time = time.time()
        print('Loading Data...')
        for i, mat_id in enumerate(ref['id']):
            path = os.path.join(self.path, f'{mat_id}.mat')
            x = spio.loadmat(path)['val'][0].astype(np.float32)
            x = np.expand_dims(x, -1)
            x = (x - x.mean()) / x.std()
            xs[i] = torch.tensor(x)
        elapsed_time = time.time() - start_time
        print(f'Loaded {ref.shape[0]} in {elapsed_time:.2f} seconds')
        y = pd.Categorical(ref['label'].values, self.CATS).codes
        sss = skms.StratifiedShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.train_pct,
            random_state=self.split_seed)
        return [
            (_Dataset(xs[train], y[train]), _Dataset(xs[val], y[val]))
            for train, val in sss.split(np.zeros(len(y)), y)
        ]

    @staticmethod
    def _collate(
            batch: List[Tuple[torch.Tensor, int]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        xs, ys = zip(*batch)
        y = torch.tensor(ys, dtype=torch.long)
        return xs, y

    def __iter__(self) -> DataLoaders:
        for train_ds, val_ds in self.datasets:
            train_dl = tud.DataLoader(
                _Augment(train_ds, self.trim_prob, self.trim_min),
                self.batch_size,
                shuffle=True,
                collate_fn=self._collate)
            val_dl = tud.DataLoader(
                val_ds, self.batch_size,
                shuffle=False,
                collate_fn=self._collate)
            yield train_dl, val_dl
