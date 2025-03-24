from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.data_prep import ImageDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, transform: bool = True, batch_size=32):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = ImageDataset(self.image_dir, transform=self.transform)

        # Compute sizes for train/val/test
        total_size = len(full_dataset)
        test_size = int(0.2 * total_size)
        val_size = int(0.1 * total_size)
        train_size = total_size - test_size - val_size

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


if __name__ == "__main__":
    image_dir = Path("data/filtered")
    dm = ImageDataModule(image_dir, batch_size=8)
    dm.setup()

    for batch in dm.train_dataloader():
        print(batch)
        break

    for batch in dm.val_dataloader():
        print(batch)
        break

    for batch in dm.test_dataloader():
        print(batch)
        break
