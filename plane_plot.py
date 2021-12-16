import matplotlib.pyplot as plt
import torch
import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
import os
from collections import OrderedDict
from torch import device
from pytorch_lightning.loggers import WandbLogger

from utils import to_tensor, to_state_dict, pick_gpus, pick_device


def get_trainer(state_dict):
    pass


@hydra.main(config_path='configs', config_name='plot_plane_toy_curve.yaml')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    n_batches = cfg.n_batches
    logger = WandbLogger(**cfg.logger, offline=True)
    trainer = Trainer(
        **cfg.trainer,
        gpus=pick_gpus(),
        limit_val_batches=n_batches
    )
    map_location = pick_device()
    cfg.model['checkpoint_path'] = to_absolute_path(cfg.model['checkpoint_path'])
    model = instantiate(cfg.model, map_location=map_location, optimizer_conf=None, metrics_conf=cfg.metrics,
                        _recursive_=False)
    assert len(model.curve.start_parameters()) == len(model.curve.inner_parameters()), 'Not 1-parametric curves?'
    datamodule = instantiate(cfg.datamodule)
    print(f'Plotting loss plane for model {type(model)}')
    single_model = instantiate(cfg.single_model, architecture=cfg.model['architecture'], optimizer_conf=None,
                               metrics_conf=cfg.metrics, _recursive_=False)
    # # print(single_model)
    # print('Single model ', list(single_model.state_dict().keys()))
    # print('Start points ', list(model.curve.curves.keys()))
    # # print('Inner params ', list(model.curve.start_parameters().keys()))
    #
    # print()
    # print('Single model ', [t.shape for t in single_model.state_dict().values()])
    # print('Start points ', [t.shape for t in model.curve.start_parameters()])
    # print('Inner params ', [t.shape for t in model.curve.inner_parameters()])
    param_names = list(single_model.state_dict().keys())
    start = OrderedDict(zip(param_names, model.curve.start_parameters()))
    end = OrderedDict(zip(param_names, model.curve.end_parameters()))
    middle = OrderedDict(zip(param_names, model.curve.inner_parameters()))
    start_tens, sizes = to_tensor(start)
    end_tens, _ = to_tensor(end)
    middle_tens, _ = to_tensor(middle)
    # first basis vector
    e_x = end_tens - start_tens
    dx = torch.linalg.norm(e_x)
    # second basis vector
    e_y_skewed = middle_tens - start_tens
    e_y = e_y_skewed - torch.inner(e_x, e_y_skewed) / dx * e_x
    dy = torch.linalg.norm(e_y)
    print(torch.linalg.norm(e_x), torch.abs(e_x).mean())


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
