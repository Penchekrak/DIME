import typing as tp
from pydoc import locate

import pytorch_lightning as pl
import torch
import wandb
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
    for param in start.values():
        param.requires_grad = True
    for param in end.values():
        param.requires_grad = True
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
            curve_loss = {}
            for t in torch.linspace(0, 1, self.n_points):
                output = self.forward(x, t)
                curve_loss[f"loss at/{t:.4f}"] = self.loss(output, y)

            self.log_dict(curve_loss, on_epoch=True, on_step=False)
            # self.logger.experiment.log({"curve loss": wandb.plot.line(loss_table, "t", "loss", title="Curve loss")})

        t = self.t_distribution.sample()

        output = self.forward(x, t)
        loss = self.loss(output, y)

        self.log('val loss', loss, on_step=True)
        self.log_dict(self.metrics(output, y), on_step=True)
        return loss

    def forward(self, x, t):
        weights = self.curve.get_point(t)
        return self.net(x, weights)

    def configure_optimizers(self):
        return instantiate(self.optimizer_conf, params=self.curve.parameters())


class MiniMaxToyTrainer(pl.LightningModule):
    def __init__(
        self,
        architecture: torch.nn.Module,
        curve: OmegaConf,
        optimizer_conf: OmegaConf,
        metrics_conf: OmegaConf,
        n_points_conf: OmegaConf,
        eps_conf: OmegaConf
    ):
        super(MiniMaxToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points_conf
        self.eps = eps_conf

        self.automatic_optimization = False

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch

        t = self.t_distribution.sample()
        ends_opt, curve_opt = self.optimizers()

        output = self.forward(x, t)
        mean_curve_loss = self.loss(output, y)
        self.log('loss', mean_curve_loss)
        curve_opt.zero_grad()
        self.manual_backward(mean_curve_loss)
        curve_opt.step()

        curve_loss = []
        for t in torch.linspace(self.eps, 1 - self.eps, self.n_points):
            output = self.forward(x, t)
            curve_loss.append(self.loss(output, y))
        curve_loss = torch.stack(curve_loss)

        max_curve_loss = torch.dot(curve_loss, torch.softmax(curve_loss, dim=0))

        output_0 = self.forward(x, 0.)
        output_1 = self.forward(x, 1.)

        adv_loss = self.loss(output_0, y) + self.loss(output_1, y) - max_curve_loss
        ends_opt.zero_grad()
        self.manual_backward(adv_loss)
        curve_opt.step()

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch

        if batch_idx == 0:
            curve_loss = {}
            for t in torch.linspace(0, 1, self.n_points):
                output = self.forward(x, t)
                curve_loss[f"loss at/{t:.4f}"] = self.loss(output, y)

            self.log_dict(curve_loss, on_epoch=True, on_step=False)
            # self.logger.experiment.log({"curve loss": wandb.plot.line(loss_table, "t", "loss", title="Curve loss")})

        t = self.t_distribution.sample()

        output = self.forward(x, t)
        loss = self.loss(output, y)

        self.log('val loss', loss, on_step=True)
        self.log_dict(self.metrics(output, y), on_step=True)
        return loss

    def forward(self, x, t):
        weights = self.curve.get_point(t)
        return self.net(x, weights)

    def configure_optimizers(self):
        curve_opt = instantiate(self.optimizer_conf, params=self.curve.parameters())
        ends_opt = instantiate(self.optimizer_conf, params=self.curve.start_parameters() +
                                                           self.curve.end_parameters())
        return ends_opt, curve_opt
