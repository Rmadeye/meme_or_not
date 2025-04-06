import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.network import MemeClassifier


@pytest.fixture
def sample_data():
    # Create dummy data for testing
    x = torch.randn(16, 3, 224, 224)  # Batch of 16 RGB images of size 224x224
    y = torch.randint(
        0, 2, (16,)
    )  # Batch of 16 labels (binary classification)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def model():
    return MemeClassifier()


def test_forward_pass(model):
    # Test the forward pass
    x = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB images of size 224x224
    y_hat = model(x)
    assert y_hat.shape == (8, 2), "Output shape mismatch in forward pass"


def test_training_step(model, sample_data):
    # Test the training step
    batch = next(iter(sample_data))
    loss = model.training_step(batch, 0)
    assert loss.requires_grad, "Loss should require gradients during training"


def test_test_step(model, sample_data):
    # Test the test step
    batch = next(iter(sample_data))
    output = model.test_step(batch, 0)
    assert (
        "test_loss" in output and "test_acc" in output
    ), "Test step output keys mismatch"
    assert isinstance(
        output["test_loss"], torch.Tensor
    ), "Test loss should be a tensor"
    assert isinstance(
        output["test_acc"], torch.Tensor
    ), "Test accuracy should be a tensor"


def test_configure_optimizers(model):
    # Test the optimizer configuration
    optimizer = model.configure_optimizers()
    assert isinstance(
        optimizer, torch.optim.Adam
    ), "Optimizer should be an instance of Adam"
