import argparse

from pathlib import Path
import pytorch_lightning as pl

from mem_or_not.network import MemeClassifier
from mem_or_not.dataloader import ImageDataModule

meme_dict = {0: "not meme", 1: "meme"}


def parse_args():
    parser = argparse.ArgumentParser(description="Meme Classifier Prediction")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory of images to predict",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    return parser.parse_args()


def predict(image_dir: Path, model_path: Path):
    model = MemeClassifier.load_from_checkpoint(model_path)
    data_module = ImageDataModule(
        predict_dir=image_dir, batch_size=1, transform=True
    )
    data_module.setup(stage="predict")

    trainer = pl.Trainer()
    predictions = trainer.predict(
        model, dataloaders=data_module.predict_dataloader()
    )

    for batch, (_, paths) in zip(
        predictions, data_module.predict_dataloader()
    ):
        batch = batch[0]  # Unpack the tensor from [[1, 0]] to [1, 0]
        for _, path in zip(batch, paths):
            percentages = batch * 100
            print(
                f"Image:{path}, Class: {meme_dict[int(batch.argmax(0))]} "
                f"Probabilities: Meme: {percentages[1]:.2f}%, "
                f"Not Meme: {percentages[0]:.2f}%"
            )


if __name__ == "__main__":
    args = parse_args()
    image_dir = Path(args.image_dir)
    model_path = Path(args.model_path)

    if not image_dir.is_dir():
        raise ValueError(f"Image directory {image_dir} does not exist.")
    if not model_path.is_file():
        raise ValueError(f"Model path {model_path} does not exist.")

    predict(image_dir, model_path)
