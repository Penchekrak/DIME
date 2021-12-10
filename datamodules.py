from typing import Optional

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = to_absolute_path(data_dir)
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.mnist_train = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)
