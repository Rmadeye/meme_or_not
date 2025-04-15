from pathlib import Path
import pytorch_lightning as pl

from mem_or_not.network import MemeClassifier
from mem_or_not.dataloader import ImageDataModule
from mem_or_not.utils import load_config


def train(config: dict):
    """
    Train the MemeClassifier model.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """
    model = MemeClassifier()
    train_cfg = config["train"]
    data_module = ImageDataModule(
        meme_dir=Path(train_cfg["meme_dir"]),
        other_dir=Path(train_cfg["other_dir"]),
        batch_size=train_cfg["batch_size"],
        transform=True,
    )
    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"], log_every_n_steps=1, fast_dev_run=True
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    # trainer.save_checkpoint(
    #     Path(train_cfg["model_dir"]) / "meme_classifier.ckpt"
    # )


if __name__ == "__main__":
    output_dir = Path("output")
    config_path = Path("hparams.yaml")
    config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    train(config)
