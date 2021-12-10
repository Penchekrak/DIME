import typing as tp

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions import Uniform

from functional_nets import FunctionalNet


class SingleToyTrainer(pl.LightningModule):
    def __init__(self, architecture: torch.nn.Module, optimizer_conf):
        super(SingleToyTrainer, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.net = instantiate(architecture)
        self.optimizer_conf = optimizer_conf

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log_dict(
            {
                'loss': loss
            }
        )
        return loss

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return instantiate(self.optimizer_conf, params=self.parameters())


class CurveToyTrainer(pl.LightningModule):
    def __init__(self, architecture: torch.nn.Module, curve: OmegaConf):
        super(CurveToyTrainer, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.net = FunctionalNet(architecture)
        self.curve = instantiate(curve)
        self.t_distribution = Uniform(0, 1)

    def training_step(self, batch: tp.Tuple[torch.Tensor, ...], batch_idx: int):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log_dict(
            {
                'loss': loss
            }
        )
        return loss

    def forward(self, x):
        t = self.t_distribution.sample()
        weights = self.curve.get_point(t)
        return self.net(x, weights)

    def configure_optimizers(self):
        return instantiate(self.optimizer_conf, params=self.curve.parameters())
