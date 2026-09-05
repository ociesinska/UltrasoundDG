from pathlib import Path

import numpy as np
from PIL import Image


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    if "A" in image.getbands():
        background = Image.new(
            "RGBA",
            image.size,
            color=(0, 0, 0, 255),
        )

        image = Image.alpha_composite(
            background,
            image.convert("RGBA"),
        )

    return np.array(image.convert("RGB"))


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return image_to_rgb_array(image)
    except OSError as error:
        raise ValueError(f"Could not load image: {path}") from error
