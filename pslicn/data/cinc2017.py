import numpy as np
import os
import pandas as pd
import requests
import scipy.io as spio
import shutil
import sklearn.model_selection as skms
import time
import torch
import torch.utils.data as tud
from typing import List, Tuple
import zipfile
from . import Data


class _Dataset(tud.Dataset):
    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        self.xs = xs
        self.y = y

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        return x, self.y[idx]


class _Augment(tud.Dataset):
    def __init__(self, ds: tud.Dataset, trim_prob: float, trim_min: float) -> None:
        self.ds = ds
        self.trim_prob = trim_prob
        self.trim_min = trim_min

    def __len__(self):
        return len(self.ds)

    def _augment(self, x):
        length: int = x.shape[0]
        trimmed: int = int(torch.randint(int(length*self.trim_min), length, ()))
        offset = torch.randint(length-trimmed, ()).item()
        return x[offset:(offset+trimmed)]

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        if self.trim_prob and (float(torch.rand(())) < self.trim_prob):
            x = self._augment(x)
        return x, y


class Cinc2017:
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
    ) -> None:
        self.path = os.path.join(base_path, self.PATH)
        self.n_folds = n_folds
        self.train_pct = train_pct
        self.split_seed = split_seed
        self.n_features = 1
        self.n_classes = len(self.CATS)

    def _download(self) -> None:
        if not os.path.exists(self.path):
            tmp_path = self.path + '_tmp'
            os.makedirs(tmp_path, exist_ok=True)
            zip_path = os.path.join(tmp_path, 'training2017.zip')
            if not os.path.exists(zip_path):
                with requests.get(self.URL, stream=True) as r:
                    start_time = time.time()
                    print('Downloading Data...')
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                    elapsed_time = time.time() - start_time
                    print(f'Downloaded in {elapsed_time:.2f} seconds')
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
            shutil.move(os.path.join(tmp_path, 'training2017'), self.path)
            shutil.rmtree(tmp_path)

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

    def __call__(self, batch_size: int, trim_prob: float, trim_min: float) -> Data:
        self._download()
        for train_ds, val_ds in self._setup():
            train_dl = tud.DataLoader(
                _Augment(train_ds, trim_prob, trim_min),
                batch_size,
                shuffle=True,
                collate_fn=self._collate)
            val_dl = tud.DataLoader(
                val_ds, batch_size,
                shuffle=False,
                collate_fn=self._collate)
            yield train_dl, val_dl
