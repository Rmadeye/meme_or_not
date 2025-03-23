import pytest
from PIL import Image
import numpy as np
import torch
from src.data_prep import load_image, transform_to_tensor, ImageDataset


@pytest.fixture
def test_image_path(tmp_path):
    # Create a dummy image for testing
    test_image_path = tmp_path / "test_image.jpg"
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(test_image).save(test_image_path)
    return test_image_path


def test_load_image(test_image_path):
    # Test if load_image correctly loads an image as a PIL Image
    loaded_image = load_image(test_image_path)
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.size == (100, 100)


def test_transform_to_tensor(test_image_path):
    # Test if transform_to_tensor correctly transforms a PIL Image to a tensor
    loaded_image = load_image(test_image_path)
    tensor = transform_to_tensor(loaded_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)  # After resizing to 256x256


@pytest.fixture
def test_image_dir(tmp_path):
    # Create a temporary directory with dummy images and a labels file
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    labels_file = image_dir / "labels.txt"
    with open(labels_file, "w") as f:
        f.write("filename,label\n")
        for i in range(5):
            img_name = f"image_{i}.jpg"
            label = bool(int(i % 2))  # Alternate labels 0 and 1
            f.write(f"{img_name},{label}\n")
            img_array = np.random.randint(
                0, 256, (100, 100, 3), dtype=np.uint8
            )
            Image.fromarray(img_array).save(image_dir / img_name)
    return image_dir


def test_image_dataset_len(test_image_dir):
    # Test the length of the dataset
    dataset = ImageDataset(image_dir=test_image_dir)
    assert len(dataset) == 5  # 5 images in the labels file


def test_image_dataset_getitem(test_image_dir):
    # Test retrieving an item from the dataset
    dataset = ImageDataset(image_dir=test_image_dir)
    image, label = dataset[0]
    assert isinstance(
        image, torch.Tensor
    )  # Image should be transformed to a tensor
    assert image.shape == (3, 256, 256)  # After resizing to 256x256
    assert label in [False, True]  # Labels are either "0" or "1"


def test_image_dataset_no_transform(test_image_dir):
    # Test dataset without applying transformations
    dataset = ImageDataset(image_dir=test_image_dir, transform=False)
    image, label = dataset[0]
    assert isinstance(image, Image.Image)  # Image should remain a PIL Image
    assert image.size == (100, 100)  # Original size
    assert label in [False, True]  # Labels are either "0" or "1"
