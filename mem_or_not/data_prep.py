from typing import Optional

from pathlib import Path
import PIL
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


def load_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as img:
            array = np.array(img)
            pil_img = Image.fromarray(array)
            return pil_img.convert("RGB")
    except PIL.UnidentifiedImageError:
        return None


class ImageDataset(Dataset):
    def __init__(
        self,
        meme_dir: Optional[Path] = None,
        other_dir: Optional[Path] = None,
        predict_dir: Optional[Path] = None,
        transform: bool = True,
        predict_mode: bool = False,
    ):
        self.transform = (
            transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            if transform
            else None
        )
        self.meme_dir = meme_dir
        self.other_dir = other_dir
        self.predict_dir = predict_dir
        self.predict_mode = predict_mode
        self.data: list = []
        self._load_data()

    def _load_data(self):
        if self.predict_mode:
            for img_path in self.predict_dir.glob("*.*"):
                self.data.append(
                    (img_path, None)
                )  # No labels for prediction mode
        else:
            for img_path in self.meme_dir.glob("*.*"):
                self.data.append((img_path, 1))  # Label 1 for meme images

            for img_path in self.other_dir.glob("*.*"):
                self.data.append((img_path, 0))  # Label 0 for other images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = load_image(img_path)
        if image is None:
            self.data.pop(idx)
            return self.__getitem__(idx)

        if self.transform:
            image = self.transform(image)

        return (
            (image, label) if not self.predict_mode else (image, str(img_path))
        )
