import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from mem_or_not.dataloader import ImageDataModule


@pytest.fixture
def mock_image_dataset():
    # Mock the ImageDataset class
    class MockDataset:
        def __len__(self):
            return 100  # Mock dataset size

    return MockDataset()


@pytest.fixture
def mock_image_data_module(mock_image_dataset, monkeypatch):
    # Mock the ImageDataset initialization
    def mock_image_dataset_init(*args, **kwargs):
        return mock_image_dataset

    monkeypatch.setattr(
        "mem_or_not.dataloader.ImageDataset", mock_image_dataset_init
    )
    image_dir = Path("data/filtered")
    return ImageDataModule(image_dir, batch_size=8)


def test_setup(mock_image_data_module):
    mock_image_data_module.setup()
    assert len(mock_image_data_module.train_dataset) == 70  # 70% of 100
    assert len(mock_image_data_module.val_dataset) == 10  # 10% of 100
    assert len(mock_image_data_module.test_dataset) == 20  # 20% of 100


def test_train_dataloader(mock_image_data_module):
    mock_image_data_module.setup()
    train_loader = mock_image_data_module.train_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 8


def test_val_dataloader(mock_image_data_module):
    mock_image_data_module.setup()
    val_loader = mock_image_data_module.val_dataloader()
    assert isinstance(val_loader, DataLoader)
    assert val_loader.batch_size == 8


def test_test_dataloader(mock_image_data_module):
    mock_image_data_module.setup()
    test_loader = mock_image_data_module.test_dataloader()
    assert isinstance(test_loader, DataLoader)
    assert test_loader.batch_size == 8
