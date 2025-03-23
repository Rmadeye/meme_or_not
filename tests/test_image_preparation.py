import pytest
from PIL import Image
from scripts.prepare_images import (
    filter_images,
    check_image_size,
    pad_image_to_minimal_size,
)


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Fixture to create a temporary directory with test images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create test images
    small_image = Image.new("RGB", (100, 100), (255, 0, 0))  # Small red image
    small_image.save(image_dir / "small_image.jpg")

    large_image = Image.new(
        "RGB", (300, 300), (0, 255, 0)
    )  # Large green image
    large_image.save(image_dir / "large_image.png")

    unsupported_image = image_dir / "unsupported.txt"
    unsupported_image.write_text("This is a text file.")

    return image_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Fixture to create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_filter_images(tmp_image_dir, tmp_output_dir):
    """Test the filter_images function."""
    filter_images(tmp_image_dir, tmp_output_dir)

    # Check that the output directory contains the expected files
    output_files = list(tmp_output_dir.iterdir())
    assert len(output_files) == 2  # Only two valid images should be processed
    assert (tmp_output_dir / "small_image.jpg").exists()
    assert (tmp_output_dir / "large_image.png").exists()


def test_check_image_size():
    """Test the check_image_size function."""
    small_image = Image.new("RGB", (100, 100), (255, 0, 0))
    large_image = Image.new("RGB", (300, 300), (0, 255, 0))

    assert not check_image_size(small_image, size=(256, 256))
    assert check_image_size(large_image, size=(256, 256))


def test_pad_image_to_minimal_size():
    """Test the pad_image_to_minimal_size function."""
    small_image = Image.new("RGB", (100, 100), (255, 0, 0))
    padded_image = pad_image_to_minimal_size(small_image, size=(256, 256))

    assert padded_image.size == (256, 256)
    assert padded_image.getpixel((0, 0)) == (0, 0, 0)  # Check padding is black
    assert padded_image.getpixel((78, 78)) == (
        255,
        0,
        0,
    )  # Check original image content
