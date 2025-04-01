from src.data_prep import ImageDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Optional


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meme_dir: Optional[Path] = None,
        other_dir: Optional[Path] = None,
        predict_dir: Optional[Path] = None,
        transform: bool = True,
        batch_size=32,
    ):
        super().__init__()
        self.meme_dir = meme_dir
        self.other_dir = other_dir
        self.predict_dir = predict_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        if self.predict_dir is None:
            full_dataset = ImageDataset(
                self.meme_dir, self.other_dir, transform=self.transform
            )

            # Compute sizes for train/val/test
            total_size = len(full_dataset)
            test_size = int(0.2 * total_size)
            val_size = int(0.1 * total_size)
            train_size = total_size - test_size - val_size

            # Split dataset
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_split(full_dataset, [train_size, val_size, test_size])
        else:
            self.predict_dataset = ImageDataset(
                predict_dir=self.predict_dir,
                transform=self.transform,
                predict_mode=True,
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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
