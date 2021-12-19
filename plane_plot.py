import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
import os
from collections import OrderedDict
import logging
from itertools import product
from torch import device
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger

from utils import to_tensor, to_state_dict, pick_gpus, pick_device
from functional_nets import FunctionalNet
from toy_model import SingleToyTrainer


def remove_pref_from_str(string: str) -> str:
    return string[string.find('.') + 1:]


def remove_pref_from_dict(state_dict):
    return dict((remove_pref_from_str(key), val) for key, val in state_dict.items())


@torch.no_grad()
@hydra.main(config_path='configs')
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    seed_everything(cfg.seed)
    n_batches = cfg.n_batches
    logger = WandbLogger(**cfg.logger, offline=True)
    trainer = Trainer(
        **cfg.trainer,
        gpus=pick_gpus(),
        limit_val_batches=n_batches,
        enable_progress_bar=False
    )
    map_location = pick_device()
    cfg.model['checkpoint_path'] = to_absolute_path(cfg.model['checkpoint_path'])
    model = instantiate(cfg.model, map_location=map_location, optimizer_conf=None, metrics_conf=cfg.metrics,
                        _recursive_=False)
    assert len(model.curve.start_parameters()) == len(model.curve.inner_parameters()), 'Not 1-parametric curves?'
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()
    single_model = SingleToyTrainer(architecture=cfg.model.architecture, optimizer_conf=cfg.optimizer,
                                    metrics_conf=cfg.metrics)
    func_net = FunctionalNet(instantiate(cfg.model.architecture))
    print(f'Plotting loss plane for model {type(model)}')
    single_model = instantiate(cfg.single_model, architecture=cfg.model['architecture'], optimizer_conf=None,
                               metrics_conf=cfg.metrics, _recursive_=False)
    param_names = list(single_model.state_dict().keys())
    start = OrderedDict(zip(param_names, model.curve.start_parameters()))
    end = OrderedDict(zip(param_names, model.curve.end_parameters()))
    middle = OrderedDict(zip(param_names, model.curve.inner_parameters()))
    start_tens, sizes = to_tensor(start)
    end_tens, _ = to_tensor(end)
    middle_tens, _ = to_tensor(middle)
    # print(torch.linalg.norm(start_tens - end_tens))
    # print(torch.linalg.norm(start_tens - middle_tens))
    # print(torch.linalg.norm(middle_tens - end_tens))
    # exit()
    # first basis vector
    e_x = end_tens - start_tens
    norm_x = torch.linalg.norm(e_x)
    # second basis vector
    e_y_skewed = middle_tens - start_tens
    e_y = e_y_skewed - torch.inner(e_x / norm_x ** 2, e_y_skewed) * e_x
    norm_y = torch.linalg.norm(e_y)
    # print(torch.linalg.norm(e_x))
    # print(torch.linalg.norm(e_y_skewed))
    # print(torch.linalg.norm(e_y))
    # print(torch.inner(e_x, e_y))
    x_middle = torch.inner(e_x, e_y_skewed) / norm_x ** 2
    y_middle = 1
    # print(x_middle, y_middle, torch.linalg.norm(e_y_skewed - x_middle * e_x - y_middle * e_y))
    # print(np.linalg.norm(middle_tens - (start_tens + x_middle*e_x + y_middle*e_y)))

    margin = cfg.margin
    n_pts = cfg.n_pts
    xs = np.linspace(-margin, 1 + margin, n_pts)
    ys = np.linspace(-margin, 1 + margin, n_pts)
    losses = np.zeros((n_pts, n_pts))
    accs = np.zeros((n_pts, n_pts))
    x_grid, y_grid = np.meshgrid(xs, ys, indexing='ij')

    for i, x in tqdm(enumerate(xs)):
        for j, y in tqdm(enumerate(ys), leave=False):
            params_tensor = start_tens + x * e_x + y * e_y
            state_dict = remove_pref_from_dict(to_state_dict(params_tensor, sizes))
            single_model.net.load_state_dict(state_dict)
            results = trainer.validate(single_model, val_dataloaders=datamodule.train_dataloader(), verbose=False)[0]
            losses[i, j] = results['val loss']
            accs[i, j] = results['Accuracy']
            # print(x, y, accs[i, j], losses[i, j])

    plt.figure(figsize=(10, 10))
    for ind, data, title in zip([1, 2], [losses, accs], ['Train losses', 'Train accuracy']):
        plt.subplot(2, 2, ind)
        plt.scatter([0, x_middle * norm_x, norm_x], [0, y_middle * norm_y, 0], c='black')
        plt.plot([0, x_middle * norm_x, norm_x], [0, y_middle * norm_y, 0], c='black')
        plt.title(title)
        plt.contourf(x_grid * float(norm_x), y_grid * float(norm_y), data, locator=ticker.LogLocator(subs='all'),
                     cmap='coolwarm')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

    if 'name' in cfg:
        name = cfg.name
    else:
        name = f'from__{cfg.model.curve.start[5:-5]}__to__{cfg.model.curve.end[5:-5]}'

    plt.savefig(to_absolute_path(f'plots/{name}.png'))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
