import typing as tp
from pydoc import locate

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, DictConfig
from torch.distributions import Uniform
from torchmetrics import MetricCollection

from curves import StateDictCurve
from functional_nets import FunctionalNet


class SingleToyTrainer(pl.LightningModule):
    def __init__(
            self,
            architecture: torch.nn.Module,
            optimizer_conf: OmegaConf,
            metrics_conf: OmegaConf
    ):
        super(SingleToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = instantiate(architecture)
        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log_dict(
            {
                'train loss': loss
            }
        )
        return loss

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log_dict(
            {
                'val loss': loss
            }
        )
        self.log_dict(self.metrics(output, y))
        return loss

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return instantiate(self.optimizer_conf, params=self.parameters())


def create_curve_from_conf(curve_conf: DictConfig):
    start = torch.load(to_absolute_path(curve_conf['start']))['state_dict']
    end = torch.load(to_absolute_path(curve_conf['end']))['state_dict']
    return StateDictCurve(start, end, curve_type=locate(curve_conf['curve_type']))


class CurveToyTrainer(pl.LightningModule):
    def __init__(
            self,
            architecture: torch.nn.Module,
            curve: OmegaConf,
            optimizer_conf: OmegaConf,
            metrics_conf: OmegaConf,
            n_points_conf: OmegaConf
    ):
        super(CurveToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points_conf

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        t = self.t_distribution.sample()
        output = self.forward(x, t)
        loss = self.loss(output, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch

        if batch_idx == 0:
            curve_loss = 0.
            for t in torch.linspace(0, 1, self.n_points):
                output = self.forward(x, t)
                curve_loss += self.loss(output, y)

            self.log('curve loss', curve_loss / self.n_points)

        t = self.t_distribution.sample()

        output = self.forward(x, t)
        loss = self.loss(output, y)

        self.log('val loss', loss)
        self.log_dict(self.metrics(output, y))
        return loss

    def forward(self, x, t):
        weights = self.curve.get_point(t)
        return self.net(x, weights)

    def configure_optimizers(self):
        return instantiate(self.optimizer_conf, params=self.curve.parameters())
