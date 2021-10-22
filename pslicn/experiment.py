from abc import ABC, abstractmethod
import argparse
from collections.abc import Mapping
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
from tqdm.auto import tqdm
from typing import Any, Generator, Optional, Tuple, Type
from .data import Data


@dataclass
class Metrics:
    loss: float


@dataclass
class Params:
    epochs: int = 150


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

    def __call__(self, model: nn.Module, batch: Any) -> torch.Tensor:
        loss, size = self._step(model, batch)
        self.total_loss += loss.item() * size
        self.total_size += size
        return loss

    @abstractmethod
    def _step(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, int]:
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
    folds: int
    rseed: int
    checkpoint: int


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
    def data(self, params: Params, folds: int, rseed: int) -> Data:
        raise NotImplementedError()

    @abstractmethod
    def model(self, params: Params) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def optimizer(self, model: nn.Module, params: Params) -> optim.Optimizer:
        raise NotImplementedError()

    @abstractmethod
    def step(self) -> Step:
        raise NotImplementedError()

    def _setup(self) -> Tuple[_Setup, Params]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--no_gpu', dest='gpu', action='store_false',
            help='Do NOT use GPU if available')
        parser.add_argument(
            '-o', '--out', default=os.path.join('results', self.default_name()),
            help='Output directory')
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
        os.makedirs(args.out, exist_ok=True)
        db = sqlalchemy.create_engine(f'sqlite+pysqlite:///{args.out}/experiments.db')
        self.metadata.create_all(db)
        if args.resume is not None:
            resume = self._resume(db, args.resume)
            if resume is None:
                raise ValueError(f'Experiment {args.resume} not found')
            elif len(args.override) > 0:
                raise ValueError('Cannot override hyperparams when resuming')
            setup = _Setup(
                db=db,
                eid=args.resume,
                gpu=args.gpu,
                out=args.out,
                folds=resume.folds,
                rseed=resume.rseed,
                checkpoint=args.checkpoint)
            return setup, resume.params
        else:
            params = OmegaConf.merge(
                OmegaConf.structured(self.default_params()),
                OmegaConf.from_cli(args.override))
            eid = self._start(db, args.folds, args.seed, params)
            setup = _Setup(
                db=db,
                eid=eid,
                gpu=args.gpu,
                out=args.out,
                folds=args.folds,
                rseed=args.seed,
                checkpoint=args.checkpoint)
            return setup, params

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

    def _start(self, db: Engine, folds: int, rseed: int, params: Mapping) -> int:
        params = dict(**params)
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
                 fold: int, epoch: int, phase: Phase,
                 metrics: Metrics) -> None:
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

    def run(self) -> None:
        s, params = self._setup()
        print(OmegaConf.to_yaml(params))
        device = torch.device('cuda' if s.gpu and torch.cuda.is_available() else 'cpu')
        for fold, (train_dl, val_dl) in enumerate(self.data(params, s.folds, s.rseed)):
            model = self.model(params)
            model.to(device)
            optimizer = self.optimizer(model, params)
            start_epoch = _load_checkpoint(s.out, s.eid, fold, model, optimizer)
            step = self.step()
            step.to(device)
            self._start_fold(s.db, s.eid, fold, start_epoch)
            for epoch in range(start_epoch, params.epochs):
                model.train()
                step.reset()
                start = datetime.now()
                with tqdm(train_dl, desc=f'Fold {fold}/Epoch {epoch}/Train') as bt:
                    for batch in bt:
                        batch = _to_device(batch, device)
                        optimizer.zero_grad()
                        loss = step(model, batch)
                        loss.backward()
                        optimizer.step()
                        bt.set_postfix(Loss=step.compute_loss())
                self._results(s.db, s.eid, start, datetime.now(), fold, epoch, Phase.Train, step.compute())
                model.eval()
                step.reset()
                start = datetime.now()
                with torch.inference_mode():
                    with tqdm(val_dl, desc=f'Fold {fold}/Epoch {epoch}/Val') as bt:
                        for batch in bt:
                            batch = _to_device(batch, device)
                            step(model, batch)
                            bt.set_postfix(Loss=step.compute_loss())
                self._results(s.db, s.eid, start, datetime.now(), fold, epoch, Phase.Val, step.compute())
                if ((epoch + 1) == params.epochs) or ((epoch + 1) % s.checkpoint == 0):
                    _save_checkpoint(s.out, s.eid, fold, epoch + 1, model, optimizer)
