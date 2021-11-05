import ast
from enum import Enum, unique
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.utils.data as tud
from typing import Dict, List, Set, Tuple
import wfdb
from . import utils, Data, DataLoaders
from .utils import _Dataset, _Augment


@unique
class Task(Enum):
    All = 1
    DiagnosticClass = 2
    DiagnosticSubclass = 3
    Diagnostic = 4
    Form = 5
    Rhythm = 6


def _extract_simple(codes: pd.Series, cats: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        { c: codes.apply(lambda cs: c in cs) for c in cats },
        index = codes.index)


def _extract_class(codes: pd.Series, cls: Dict[str, List[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        { c : codes.apply(lambda cs: not cs.isdisjoint(items)) for c, items in cls.items() },
        index = codes.index)


class Ptbxl(Data):
    URL = 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip'
    PATH = 'ptbxl'
    INNER_NAME = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
    N_FEATURES = 12

    def __init__(
            self,
            task: Task,
            base_path: str = 'data',
            high_res: bool = False,
            batch_size: int = 32,
            trim_prob: float = 0.9,
            trim_min: float = 0.5,
    ) -> None:
        self.task = task
        self.path = os.path.join(base_path, self.PATH)
        self.high_res = high_res
        self.batch_size = batch_size
        self.trim_prob = trim_prob
        self.trim_min = trim_min
        scp_path = os.path.join(self.path, 'scp_statements.csv')
        self.scp = pd.read_csv(scp_path, index_col = 0)
        utils.download_and_unzip(self.URL, self.INNER_NAME, self.path)
        self.train_ds, self.val_ds, self.cats = self._setup()

    @property
    def n_features(self) -> int:
        return self.N_FEATURES

    def _extract_labels(self, db: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        scp = self.scp
        codes = db['scp_codes'].apply(lambda s: set(ast.literal_eval(s).keys()))
        if self.task == Task.All:
            cats = list(scp.index)
            return _extract_simple(codes, cats), cats
        elif self.task == Task.DiagnosticClass:
            cls = {c: list(scpc.index) for c, scpc in scp.groupby('diagnostic_class')}
            return _extract_class(codes, cls), list(cls.keys())
        elif self.task == Task.DiagnosticSubclass:
            cls = {c: list(scpc.index) for c, scpc in scp.groupby('diagnostic_subclass')}
            return _extract_class(codes, cls), list(cls.keys())
        elif self.task == Task.Diagnostic:
            cats = list(scp.index[~scp['diagnostic'].isna()])
            return _extract_simple(codes, cats), cats
        elif self.task == Task.Form:
            cats = list(scp.index[~scp['form'].isna()])
            return _extract_simple(codes, cats), cats
        elif self.task == Task.Rhythm:
            cats = list(scp.index[~scp['rhythm'].isna()])
            return _extract_simple(codes, cats), cats

    def _setup(self) -> Tuple[_Dataset, _Dataset, Set[str]]:
        db_path = os.path.join(self.path, 'ptbxl_database.csv')
        db = pd.read_csv(db_path, index_col='ecg_id')
        fname_col = 'filename_hr' if self.high_res else 'filename_lr'
        xs = np.empty(db.shape[0], dtype=np.object)
        start_time = time.time()
        for i, fname in enumerate(db[fname_col]):
            x, _ = wfdb.rdsamp(os.path.join(self.path, fname))
            x = x.astype(np.float32)
            x = (x - x.mean()) / x.std()
            xs[i] = torch.tensor(x)
        elapsed_time = time.time() - start_time
        print(f'Loaded {db.shape[0]} in {elapsed_time:.2f} seconds')
        y, cats = self._extract_labels(db)
        y = y.values
        train = y.any(axis=1) & (db['strat_fold'] <= 8)
        train_ds = _Dataset(xs[train], torch.tensor(y[train]))
        val = y.any(axis=1) & (db['strat_fold'] == 9)
        val_ds = _Dataset(xs[val], torch.tensor(y[val]))
        return train_ds, val_ds, cats

    @staticmethod
    def _collate(
            batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        xs, ys = zip(*batch)
        y = torch.stack(ys, dim=0)
        return xs, y

    def __iter__(self) -> DataLoaders:
        train_dl = tud.DataLoader(
            _Augment(self.train_ds, self.trim_prob, self.trim_min),
            self.batch_size,
            shuffle=True,
            collate_fn=self._collate)
        val_dl = tud.DataLoader(
            self.val_ds, self.batch_size,
            shuffle=False,
            collate_fn=self._collate)
        yield train_dl, val_dl
