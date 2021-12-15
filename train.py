import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger

from utils import pick_gpus


@hydra.main(config_path='configs')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    logger = WandbLogger(**cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = ModelCheckpoint(**cfg.checkpoint, save_weights_only=False)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=checkpoint,
        gpus=pick_gpus()
        # plugins=DDPPlugin(find_unused_parameters=True)
    )
    model = instantiate(cfg.model, optimizer_conf=cfg.optimizer, metrics_conf=cfg.metrics, _recursive_=False)
    datamodule = instantiate(cfg.datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
