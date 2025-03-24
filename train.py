from pathlib import Path
import pytorch_lightning as pl

from src.network import MemeClassifier
from src.dataloader import ImageDataModule


def train(data_dir: Path):
    """
    Train the MemeClassifier model.

    Args:
        data_dir (Path): Path to the directory containing the training data.
        output_dir (Path): Path to the directory where the model will be saved.
    """
    model = MemeClassifier()
    data_module = ImageDataModule(
        image_dir=data_dir,
        transform=True,
        batch_size=8,
    )
    trainer = pl.Trainer(
        max_epochs=5,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    output_dir = Path("output")
    data_dir = Path("data/filtered")

    output_dir.mkdir(parents=True, exist_ok=True)

    train(data_dir)
