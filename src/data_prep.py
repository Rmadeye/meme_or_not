from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def load_image(image_path: Path) -> Image.Image:
    if image_path.suffix in [".jpg", ".jpeg", ".png"]:
        with Image.open(image_path) as img:
            array = np.array(img)
            pil_img = Image.fromarray(array)
            return pil_img
    elif image_path.suffix == ".txt":
        return None
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")


def transform_to_tensor(image: Image.Image) -> torch.Tensor:
    return transform(image)


class ImageDataset(Dataset):
    def __init__(self, image_dir: Path, transform: bool = True):
        self.transform = transform
        self.image_dir = image_dir
        self.data = []

        with open(image_dir / "labels.txt", "r") as f:
            for line in f.readlines()[1:]:
                filename, label = line.strip().split(",")
                self.data.append((filename, bool(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_full_path = self.image_dir / img_path

        image = load_image(img_full_path)
        if self.transform:
            image = transform_to_tensor(image)

        return image, label
