from abc import ABC, abstractmethod
import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import io
import numpy as np
from omegaconf import OmegaConf
import os
import re
import sqlalchemy
from sqlalchemy import Column, ForeignKey, MetaData, Table
from sqlalchemy.engine import Engine
import sqlalchemy.types as sqlt
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tmet
from typing import Any, Generator, Optional, Tuple, Type
from .data import Data


@dataclass
class Metrics:
    loss: float

    def summary(self) -> str:
        return f'Loss={self.loss:.3f}'


@dataclass
class Params:
    epochs: int = 200


class Phase(Enum):
    Train = 0
    Val = 1


class Step(nn.Module, ABC):
    def reset(self) -> None:
        def reset_child(m: nn.Module) -> None:
            if isinstance(m, tmet.Metric):
                m.reset()
        self.apply(reset_child)
        self.total_loss = 0.0
        self.total_size = 0

    def __call__(self, model: nn.Module, batch: Any, phase: Phase) -> torch.Tensor:
        loss, size = self._step(model, batch, phase)
        self.total_loss += loss.item() * size
        self.total_size += size
        return loss

    @abstractmethod
    def _step(self, model: nn.Module, batch: Any, phase: Phase) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError()

    def compute_loss(self) -> float:
        return self.total_loss / self.total_size

    @abstractmethod
    def compute(self) -> Metrics:
        raise NotImplementedError()


@dataclass(frozen=True)
class _Resume:
    folds: int
    rseed: int
    params: Params


@dataclass(frozen=True)
class _Setup:
    db: Engine
    eid: int
    gpu: bool
    out: str
    data: str
    folds: int
    rseed: int
    checkpoint: int
    params: Params


class _NumpyType(sqlt.TypeDecorator):
    impl = sqlt.LargeBinary

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        out = io.BytesIO()
        np.save(out, value)
        return out.getvalue()

    def process_literal_param(self, value, dialect):
        return self.process_bind_param(value, dialect)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        buffer = io.BytesIO(value)
        return np.load(buffer)


def _columns(cls: Any) -> Generator[Tuple[str, sqlt.TypeEngine], None, None]:
    for field in dataclasses.fields(cls):
        if field.type == bool:
            yield field.name, sqlt.Boolean
        elif field.type == int:
            yield field.name, sqlt.Integer
        elif field.type == float:
            yield field.name, sqlt.Float
        elif field.type == str:
            yield field.name, sqlt.Text
        elif issubclass(field.type, Enum):
            yield field.name, sqlt.Enum(field.type)
        elif issubclass(field.type, np.ndarray):
            yield field.name, _NumpyType


_CHECKPOINT_RE = re.compile(r'expt(\d+)_fold(\d+)_epochs(\d+)\.pt')


def _load_checkpoint(
    out: str, expt: int, fold: int,
    model: nn.Module, optimizer: optim.Optimizer
) -> int:
    best_fname = None
    best_epoch = -1
    for fname in os.listdir(out):
        m = _CHECKPOINT_RE.match(fname)
        if m is not None:
            if int(m.group(1))==expt and int(m.group(2))==fold:
                epoch = int(m.group(3))
                if epoch > best_epoch:
                    best_fname = fname
                    best_epoch = epoch
    if best_fname is None:
        return 0
    state = torch.load(os.path.join(out, best_fname))
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    return best_epoch


def _save_checkpoint(
    out: str, expt: int, fold: int, epochs: int,
    model: nn.Module, optimizer: optim.Optimizer
) -> None:
    state = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    path = os.path.join(out, f'expt{expt}_fold{fold}_epochs{epochs}.pt')
    torch.save(state, path)


def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, tuple):
        return tuple(_to_device(b, device) for b in batch)
    elif isinstance(batch, list):
        return [_to_device(b, device) for b in batch]
    else:
        raise ValueError()


class Experiment(ABC):
    def __init__(self) -> None:
        self.metadata = MetaData()
        self.experiment_table = Table(
            'experiments', self.metadata,
            Column('id', sqlt.Integer, primary_key=True),
            Column('folds', sqlt.Integer, nullable=False),
            Column('rseed', sqlt.Integer, nullable=False),
            Column('start', sqlt.DateTime, nullable=False),
            *(Column(fname, ftype) for fname, ftype in _columns(type(self.default_params()))),
            extend_existing=True)
        self.results_table = Table(
            'results', self.metadata,
            Column('experiment_id', ForeignKey('experiments.id'), nullable=False),
            Column('start', sqlt.DateTime, nullable=False),
            Column('finish', sqlt.DateTime, nullable=False),
            Column('fold', sqlt.Integer, nullable=False),
            Column('epoch', sqlt.Integer, nullable=False),
            Column('phase', sqlt.Enum(Phase), nullable=False),
            *(Column(fname, ftype) for fname, ftype in _columns(self.metrics_class())),
            extend_existing=True)

    @abstractmethod
    def default_params(self) -> Params:
        raise NotImplementedError()

    @abstractmethod
    def metrics_class(self) -> Type[Metrics]:
        raise NotImplementedError()

    @abstractmethod
    def default_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def data(self, base_path: str, params: Params, folds: int, rseed: int) -> Data:
        raise NotImplementedError()

    @abstractmethod
    def model(self, params: Params, data: Data) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def optimizer(self, model: nn.Module, params: Params) -> optim.Optimizer:
        raise NotImplementedError()

    @abstractmethod
    def step(self, data: Data) -> Step:
        raise NotImplementedError()

    def _setup_out(self, out:Optional[str] = None) -> Tuple[str, Engine]:
        if out is None:
            out = os.path.join('results', self.default_name())
        os.makedirs(out, exist_ok=True)
        db = sqlalchemy.create_engine(f'sqlite+pysqlite:///{out}/experiments.db')
        self.metadata.create_all(db)
        return out, db

    def setup_new(
            self,
            out:Optional[str] = None,
            data:str = 'data',
            gpu: bool = True,
            folds: int = 5,
            seed: int = 1234,
            checkpoint: int = 25,
            params: Optional[Params] = None) -> _Setup:
        out, db = self._setup_out(out)
        if params is None:
            params = self.default_params()
        eid = self._start(db, folds, seed, params)
        return _Setup(
            db=db,
            eid=eid,
            gpu=gpu,
            out=out,
            data=data,
            folds=folds,
            rseed=seed,
            checkpoint=checkpoint,
            params=params)

    def setup_resume(
            self,
            eid:int,
            out:Optional[str] = None,
            data:str = 'data',
            gpu:bool = True,
            checkpoint:int = 25) -> _Setup:
        out, db = self._setup_out(out)
        resume = self._resume(db, eid)
        if resume is None:
            raise ValueError(f'Experiment {eid} not found')
        return _Setup(
            db=db,
            eid=eid,
            gpu=gpu,
            out=out,
            data=data,
            folds=resume.folds,
            rseed=resume.rseed,
            checkpoint=checkpoint,
            params=resume.params)

    def _setup_from_args(self) -> _Setup:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--no_gpu', dest='gpu', action='store_false',
            help='Do NOT use GPU if available')
        parser.add_argument(
            '-o', '--out', default=os.path.join('results', self.default_name()),
            help='Output directory')
        parser.add_argument(
            '--data', default='data',
            help='Data directory')
        parser.add_argument(
            '--resume', type=int, required=False,
            help='Experiment ID to resume')
        parser.add_argument(
            '--folds', type=int, default=5,
            help='# of folds for CV')
        parser.add_argument(
            '--seed', type=int, default=1234,
            help='Random seed for CV splits')
        parser.add_argument(
            '--checkpoint', type=int, default=25,
            help='Frequency of checkpointing, in epochs')
        parser.add_argument(
            'override', nargs='*',
            help='Hyperparameter overrides')
        args = parser.parse_args()
        if args.resume is not None:
            if len(args.override) > 0:
                raise ValueError('Cannot override hyperparams when resuming')
            return self.setup_resume(args.resume, args.out, args.data, args.gpu, args.checkpoint)
        else:
            params = OmegaConf.merge(
                OmegaConf.structured(self.default_params()),
                OmegaConf.from_cli(args.override))
            return self.setup_new(
                out=args.out,
                data=args.data,
                gpu=args.gpu,
                folds=args.folds,
                seed=args.seed,
                checkpoint=args.checkpoint,
                params=OmegaConf.to_object(params))

    def _resume(self, db: Engine, expt_id: int) -> Optional[_Resume]:
        with db.begin() as conn:
            res = conn.execute(
                sqlalchemy.select(self.experiment_table) \
                    .where(self.experiment_table.c.id == expt_id))
            row = res.first()
        if row is None:
            return None
        params = self.default_params()
        for k in row.keys():
            if hasattr(params, k):
                setattr(params, k, row[k])
        return _Resume(
            folds=row.folds,
            rseed=row.rseed,
            params=params)

    def _start(self, db: Engine, folds: int, rseed: int, params: Params) -> int:
        params = dataclasses.asdict(params)
        for k, v in params.items():
            if isinstance(v, Enum):
                params[k] = v.name
        with db.begin() as conn:
            res = conn.execute(
                self.experiment_table.insert() \
                    .values(folds=folds, rseed=rseed, start=datetime.now(), **params))
            return res.inserted_primary_key[0]

    def _start_fold(self, db: Engine, expt_id: int, fold: int, start_epoch: int) -> None:
        if start_epoch > 0:
            with db.begin() as conn:
                conn.execute(
                    self.results_table.delete() \
                        .where(self.results_table.c.experiment_id == expt_id) \
                        .where(self.results_table.c.fold == fold) \
                        .where(self.results_table.c.epoch >= start_epoch))

    def _results(self,
                 db: Engine, expt_id: int,
                 start: datetime, finish: datetime,
                 fold: int, epoch: int, phase: Phase, batches: int,
                 metrics: Metrics) -> None:
        elapsed = (finish - start).total_seconds()
        perbatch = elapsed / batches
        print(f'Fold {fold}/Epoch {epoch}/{phase.name}: {batches} batches in {elapsed:.1f} s ({perbatch:.3f} s/batch) {metrics.summary()}')
        with db.begin() as conn:
            conn.execute(
                self.results_table.insert().values(
                    experiment_id=expt_id,
                    start=start,
                    finish=finish,
                    fold=fold,
                    epoch=epoch,
                    phase=phase.name,
                    **dataclasses.asdict(metrics)))

    def run(self, s:_Setup) -> None:
        print(OmegaConf.to_yaml(s.params))
        device = torch.device('cuda' if s.gpu and torch.cuda.is_available() else 'cpu')
        data = self.data(s.data, s.params, s.folds, s.rseed)
        for fold, (train_dl, val_dl) in enumerate(data):
            model = self.model(s.params, data)
            model.to(device)
            optimizer = self.optimizer(model, s.params)
            start_epoch = _load_checkpoint(s.out, s.eid, fold, model, optimizer)
            step = self.step(data)
            step.to(device)
            self._start_fold(s.db, s.eid, fold, start_epoch)
            for epoch in range(start_epoch, s.params.epochs):
                model.train()
                step.reset()
                start = datetime.now()
                for batch in train_dl:
                    batch = _to_device(batch, device)
                    optimizer.zero_grad()
                    loss = step(model, batch, Phase.Train)
                    loss.backward()
                    optimizer.step()
                self._results(s.db, s.eid, start, datetime.now(), fold, epoch, Phase.Train, len(train_dl), step.compute())
                model.eval()
                step.reset()
                start = datetime.now()
                with torch.inference_mode():
                    for batch in val_dl:
                        batch = _to_device(batch, device)
                        step(model, batch, Phase.Val)
                self._results(s.db, s.eid, start, datetime.now(), fold, epoch, Phase.Val, len(val_dl), step.compute())
                if ((epoch + 1) == s.params.epochs) or ((epoch + 1) % s.checkpoint == 0):
                    _save_checkpoint(s.out, s.eid, fold, epoch + 1, model, optimizer)

    def main(self):
        s = self._setup_from_args()
        self.run(s)
