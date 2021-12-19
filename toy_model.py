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
from utils import to_device, distance, unpack_ends


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
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log('val loss', loss)
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
    # if "checkpoint" in curve_conf:
    #     curve_state_dict = torch.load(to_absolute_path(curve_conf['path']), map_location=map_location)['state_dict']
    #     start, end = unpack_ends(curve_state_dict)
    # else:
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
            n_points: int = 10,
            freeze_start: bool = False,
            freeze_end: bool = False
    ):
        super(CurveToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve, freeze_start, freeze_end, map_location=self.device)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        t = self.t_distribution.sample()
        w = self.curve.get_point(t)
        output = self.forward(x, w)
        loss = self.loss(output, y)
        self.log('loss', loss)
        return loss

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
        return instantiate(self.optimizer_conf, params=self.curve.parameters())


class CurveEndsToyTrainer(pl.LightningModule):
    def __init__(
            self,
            architecture: torch.nn.Module,
            curve: OmegaConf,
            optimizer_conf: OmegaConf,
            metrics_conf: OmegaConf,
            n_points: int = 10,
            freeze_start: bool = False,
            freeze_end: bool = False,
            C: float = 1.0
    ):
        super(CurveEndsToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve,
                                                            freeze_start,
                                                            freeze_end,
                                                            map_location=self.device)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points
        self.C = C

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        w1 = self.curve.get_point(0.)
        w2 = self.curve.get_point(1.)
        output1 = self.forward(x, w1)
        output2 = self.forward(x, w2)
        l1 = self.loss(output1, y)
        l2 = self.loss(output2, y)
        d = distance(w1, w2)
        self.log('loss/1/train', l1)
        self.log('loss/2/train', l2)
        self.log('distance', d)
        return l1 + l2 + self.C / (1 + d)

    def validation_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch

        if batch_idx == 0:
            log_loss_along_curve(batch, self)

        w1 = self.curve.get_point(0.)
        w2 = self.curve.get_point(1.)
        output1 = self.forward(x, w1)
        output2 = self.forward(x, w2)
        l1 = self.loss(output1, y)
        l2 = self.loss(output2, y)
        self.log('loss/1/val', l1)
        self.log('loss/2/val', l2)
        self.log_dict(self.metrics(output1, y), on_step=True)
        self.log_dict(self.metrics(output2, y), on_step=True)
        return l1 + l2

    def forward(self, x, weights):
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
            freeze_end: bool = False,
            C: float = 1.0,
            adv_period: int = 1,
            max_curve_loss_ratio: float = 0.
    ):
        super(MiniMaxToyTrainer, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.net = FunctionalNet(instantiate(architecture))

        self.optimizer_conf = optimizer_conf
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

        self.curve: StateDictCurve = create_curve_from_conf(curve,
                                                            freeze_start,
                                                            freeze_end,
                                                            map_location=self.device)
        self.t_distribution = Uniform(0, 1)
        self.n_points = n_points
        self.eps = eps
        self.C = C
        self.adv_period = adv_period
        self.max_curve_loss_ratio = max_curve_loss_ratio

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

        w0 = self.curve.get_point(0.)
        w1 = self.curve.get_point(1.)
        output_0 = self.forward(x, w0)
        output_1 = self.forward(x, w1)
        l0 = self.loss(output_0, y)
        l1 = self.loss(output_1, y)
        d = distance(w0, w1)

        if batch_idx % self.adv_period == 0:
            curve_loss = []
            for t in torch.linspace(self.eps, 1 - self.eps, self.n_points):
                w = self.curve.get_point(t)
                output = self.forward(x, w)
                curve_loss.append(self.loss(output, y))
            curve_loss = torch.stack(curve_loss)

            max_curve_loss = torch.dot(curve_loss, torch.softmax(curve_loss, dim=0))
            mean_curve_loss = torch.mean(curve_loss, dim=0)
            self.log("loss/max_curve", max_curve_loss)
            if max_curve_loss > self.max_curve_loss_ratio * max(l0, l1):
                mean_curve_loss = 0.

            adv_loss = l0 + l1 - mean_curve_loss + self.C / (1 + d)

            self.log("loss/w0", l0)
            self.log("loss/w1", l1)
            self.log("loss/adv", adv_loss)
            self.log("distance", d)

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
