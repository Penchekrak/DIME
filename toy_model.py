import typing as tp
from pydoc import locate

import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, DictConfig
from torch.distributions import Uniform
from torchmetrics import MetricCollection

from curves import StateDictCurve
from functional_nets import FunctionalNet
from utils import to_device, distance


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


def create_curve_from_conf(curve_conf: DictConfig,
                           freeze_start: bool = False,
                           freeze_end: bool = False,
                           map_location: tp.Union[str, torch.DeviceObjType] = 'cpu'):
    start = torch.load(to_absolute_path(curve_conf['start']), map_location=map_location)['state_dict']
    end = torch.load(to_absolute_path(curve_conf['end']), map_location=map_location)['state_dict']
    return StateDictCurve(start, end,
                          curve_type=locate(curve_conf['curve_type']),
                          freeze_start=freeze_start,
                          freeze_end=freeze_end)


@torch.no_grad()
def log_loss_along_curve(batch, module):
    x, y = batch
    curve_loss = []
    ts = torch.linspace(0, 1, module.n_points)
    for t in ts:
        w = module.curve.get_point(t)
        output = module.forward(x, w)
        curve_loss.append(module.loss(output, y).item())
    fig = go.Figure(data=go.Scatter(x=ts, y=curve_loss))
    module.logger.experiment.log({"curve loss": fig})


class CurveToyTrainer(pl.LightningModule):
    def __init__(
            self,
            architecture: torch.nn.Module,
            curve: OmegaConf,
            optimizer_conf: OmegaConf,
            metrics_conf: OmegaConf,
            n_points: int = 10
    ):
        super(CurveToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve, map_location=self.device)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points

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
            log_loss_along_curve(batch, self)
        #     curve_loss = {}
        #     for t in torch.linspace(0, 1, self.n_points):
        #         output = self.forward(x, t)
        #         curve_loss[f"loss at/{t:.4f}"] = self.loss(output, y)
        #
        #     self.log_dict(curve_loss, on_epoch=True, on_step=False)
        #     self.logger.experiment.log({"curve loss": wandb.plot.line(loss_table, "t", "loss", title="Curve loss")})

        t = self.t_distribution.sample()

        output = self.forward(x, t)
        loss = self.loss(output, y)

        self.log('val loss', loss, on_step=True)
        self.log_dict(self.metrics(output, y), on_step=True)
        return loss

    def forward(self, x, t):
        weights = self.curve.get_point(t)
        to_device(weights, self.device)
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
            eps: float = 0.1,
            n_points: int = 10,
            freeze_start: bool = False,
            freeze_end: bool = False
    ):
        super(MiniMaxToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve, freeze_start, freeze_end, map_location=self.device)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points
        self.eps = eps

        self.automatic_optimization = False

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        ends_opt, curve_opt = self.optimizers()

        t = self.t_distribution.sample()
        w = self.curve.get_point(t)

        output = self.forward(x, w)
        mean_curve_loss = self.loss(output, y)
        self.log('loss/mean_curve', mean_curve_loss)

        curve_opt.zero_grad()
        self.manual_backward(mean_curve_loss)
        curve_opt.step()

        curve_loss = []
        for t in torch.linspace(self.eps, 1 - self.eps, self.n_points):
            w = self.curve.get_point(t)
            output = self.forward(x, w)
            curve_loss.append(self.loss(output, y))
        curve_loss = torch.stack(curve_loss)

        max_curve_loss = torch.dot(curve_loss, torch.softmax(curve_loss, dim=0))

        w0 = self.curve.get_point(0.)
        w1 = self.curve.get_point(1.)
        output_0 = self.forward(x, w0)
        output_1 = self.forward(x, w1)
        loss_0 = self.loss(output_0, y)
        loss_1 = self.loss(output_1, y)
        dist = distance(w0, w1)

        adv_loss = loss_0 + loss_1 - max_curve_loss

        self.log("loss/w0", loss_0)
        self.log("loss/w1", loss_1)
        self.log("loss/max_curve", max_curve_loss)
        self.log("loss/adv", adv_loss)
        self.log("distance", dist)

        ends_opt.zero_grad()
        self.manual_backward(adv_loss)
        ends_opt.step()

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch

        if batch_idx == 0:
            log_loss_along_curve(batch, self)

        t = self.t_distribution.sample()
        w = self.curve.get_point(t)

        output = self.forward(x, w)
        loss = self.loss(output, y)

        self.log('val loss', loss, on_step=True)
        self.log_dict(self.metrics(output, y), on_step=True)
        return loss

    def forward(self, x, weights):
        to_device(weights, self.device)
        return self.net(x, weights)


    def configure_optimizers(self):
        curve_opt = instantiate(self.optimizer_conf, params=self.curve.inner_parameters())
        ends_opt = instantiate(self.optimizer_conf, params=self.curve.start_parameters() +
                                                           self.curve.end_parameters())
        return ends_opt, curve_opt
