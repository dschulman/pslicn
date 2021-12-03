import numpy as np
import os
import requests
import shutil
import time
import torch
import torch.utils.data as tud
from typing import Union
import zipfile


def download_and_unzip(url: str, inner_path: str, path: str) -> None:
    if not os.path.exists(path):
        tmp_path = path + '_tmp'
        os.makedirs(tmp_path, exist_ok=True)
        zip_path = os.path.join(tmp_path, 'download.zip')
        if not os.path.exists(zip_path):
            start_time = time.time()
            print('Downloading Data...')
            with requests.get(url, stream=True) as r:
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            elapsed_time = time.time() - start_time
            print(f'Downloaded in {elapsed_time:.2f} seconds')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        shutil.move(os.path.join(tmp_path, inner_path), path)
        shutil.rmtree(tmp_path)


class _Dataset(tud.Dataset):
    def __init__(self, xs: np.ndarray, y: Union[np.ndarray, torch.Tensor]) -> None:
        self.xs = xs
        self.y = y

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        return x, self.y[idx]


def _augment(x: torch.Tensor, trim_min: float) -> torch.Tensor:
    length: int = x.shape[0]
    trimmed: int = int(torch.randint(int(length * trim_min), length, ()))
    offset = torch.randint(length - trimmed, ()).item()
    return x[offset:(offset + trimmed)]


class _Augment(tud.Dataset):
    def __init__(self, ds: tud.Dataset, trim_prob: float, trim_min: float) -> None:
        self.ds = ds
        self.trim_prob = trim_prob
        self.trim_min = trim_min

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        if self.trim_prob and (float(torch.rand(())) < self.trim_prob):
            x = _augment(x, self.trim_min)
        return x, y


class _TestAugment(tud.Dataset):
    def __init__(self, ds: tud.Dataset, n_aug: int, trim_min: float) -> None:
        self.ds = ds
        self.n_aug = n_aug
        self.trim_min = trim_min

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        xs = [_augment(x, self.trim_min) for _ in range(self.n_aug)]
        return xs, y
