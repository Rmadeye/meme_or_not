from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from mem_or_not.metrics import save_evaluation_metrics
from mem_or_not.network import MemeClassifier
from mem_or_not.dataloader import ImageDataModule
from mem_or_not.utils import load_config


def train(config: dict):
    """
    Train the MemeClassifier model.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """
    train_cfg = config["train"]
    network_cfg = config["network"]
    model = MemeClassifier(**network_cfg)
    data_module = ImageDataModule(
        meme_dir=Path(train_cfg["meme_dir"]),
        other_dir=Path(train_cfg["other_dir"]),
        batch_size=train_cfg["batch_size"],
        transform=True,
    )
    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=train_cfg["early_stopping"]
            )
        ],
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    trainer.save_checkpoint(Path(train_cfg["model_dir"]) / "model.ckpt")
    results = model.test_outputs[0]

    save_evaluation_metrics(results, train_cfg)
    return


if __name__ == "__main__":
    output_dir = Path("output")
    config_path = Path("hparams.yaml")
    config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    train(config)
