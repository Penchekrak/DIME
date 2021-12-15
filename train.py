import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count

@hydra.main(config_path='configs')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logger = WandbLogger(**cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = ModelCheckpoint(**cfg.checkpoint, save_weights_only=True)
    n_devices = device_count()
    if n_devices == 0:
        gpus = None
    elif n_devices == 2:
        # to use only the second GPU on statml3
        gpus = [1]
    else:
        gpus = -1
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=checkpoint,
        gpus=gpus
        # plugins=DDPPlugin(find_unused_parameters=True)
    )
    model = instantiate(cfg.model, optimizer_conf=cfg.optimizer, metrics_conf=cfg.metrics, _recursive_=False)
    datamodule = instantiate(cfg.datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
