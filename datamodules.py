from typing import Optional

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "path/to/dir",
                 batch_size: int = 32,
                 transform=transforms.ToTensor(),
                 num_workers=4):
        super().__init__()
        self.data_dir = to_absolute_path(data_dir)
        self.batch_size = batch_size
        self.transform = transform
        self.train_dataset = None
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class MNISTDataModule(DataModule):
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MNIST(self.data_dir,
                                   train=True,
                                   download=True,
                                   transform=self.transform)
        self.val_dataset = MNIST(self.data_dir,
                                 train=False,
                                 download=True,
                                 transform=self.transform)


class CIFAR10DataModule(DataModule):
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CIFAR10(self.data_dir,
                                     train=True,
                                     download=True,
                                     transform=self.transform)


class FashoinMNISTDataModule(DataModule):
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = FashionMNIST(self.data_dir,
                                          train=True,
                                          download=True,
                                          transform=self.transform)
