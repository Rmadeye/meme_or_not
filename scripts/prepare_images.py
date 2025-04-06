from pathlib import Path
from typing import Tuple

from PIL import Image
import pillow_heif
from tqdm import tqdm
import argparse


def filter_images(image_dir: Path, output_dir: Path):
    """Filter unsupported images and save supported images to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(
        image_dir.glob("*"),
        desc="Preparing images",
        total=len(list(image_dir.glob("*"))),
    ):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            suffix = image_path.suffix.lower()
            with Image.open(image_path) as img:
                if check_image_size(img, size=(256, 256)):
                    # save the img with suffix lowercased
                    img.save(output_dir / image_path.with_suffix(suffix).name)
                else:
                    padded_img = pad_image_to_minimal_size(
                        img, size=(256, 256)
                    )
                    padded_img.save(
                        output_dir / image_path.with_suffix(suffix).name
                    )
        elif image_path.suffix.lower() == ".heic":
            with open(image_path, "rb"):
                heif_file = pillow_heif.open_heif(image_path)
                pil_img = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                if check_image_size(pil_img):
                    pil_img.save(
                        output_dir / image_path.with_suffix(".png").name
                    )
        else:
            continue


def check_image_size(
    image: Image.Image, size: Tuple[int, int] = (256, 256)
) -> bool:
    """Check if the image size is larger than min_size."""
    width, height = image.size
    if width < size[0] or height < size[1]:
        return False
    return True


def pad_image_to_minimal_size(
    image: Image.Image, size: Tuple[int, int] = (256, 256)
) -> Image:
    """Pad the image with black borders if it's smaller than the given size."""
    width, height = image.size
    target_width, target_height = size

    # Ensure only smaller images are padded
    new_width = max(width, target_width)
    new_height = max(height, target_height)

    # Calculate padding
    pad_left = (new_width - width) // 2
    pad_top = (new_height - height) // 2

    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    padded_image.paste(image, (pad_left, pad_top))

    return padded_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare images for processing."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where processed images will be saved.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    filter_images(image_dir, output_dir)
