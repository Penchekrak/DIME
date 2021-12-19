import typing as tp
from copy import deepcopy

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from torchmetrics import MetricCollection


class EnsembleEvaluator(pl.LightningModule):
    def __init__(
            self,
            checkpoints: tp.Iterable[str],
            architecture: tp.Optional[tp.Union[pl.LightningModule, torch.nn.Module]] = None,
            weights: tp.Optional[tp.Iterable[float]] = None,
            metrics_conf: tp.Optional[OmegaConf] = None,
            optimizer_conf=None,
            *args, **kwargs
    ):
        super(EnsembleEvaluator, self).__init__(*args, **kwargs)
        self.models = []
        for ckpt in checkpoints:
            m = instantiate(architecture)
            m.load_state_dict(torch.load(to_absolute_path(ckpt))['state_dict'])
            self.models.append(m)
        if weights is None:
            self.weights = torch.ones(len(self.models)) / len(self.models)
        else:
            self.weights = torch.tensor(weights)
            self.weights /= self.weights.sum()
        self.metrics: MetricCollection = MetricCollection([instantiate(metric) for metric in metrics_conf])

    def forward(self, x: torch.Tensor):
        predictions = self.models[0](x) * self.weights[0]
        for i in range(1, len(self.models)):
            predictions += self.models[i](x) * self.weights[i]
        return predictions

    def predict(self, x: torch.Tensor):
        predictions = self.forward(x)
        return torch.argmax(predictions, -1)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        self.log_dict(self.metrics(outputs, y), on_step=False, on_epoch=True)

    def training_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass
