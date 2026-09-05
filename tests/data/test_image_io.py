from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ultrasound_dg.data.image_io import image_to_rgb_array, load_rgb_image


def test_image_to_rgb_array_composites_transparency_on_black() -> None:
    rgba = np.array(
        [
            [[255, 255, 255, 0], [10, 20, 30, 255]],
        ],
        dtype=np.uint8,
    )

    rgb = image_to_rgb_array(Image.fromarray(rgba, mode="RGBA"))

    assert rgb.dtype == np.uint8
    assert rgb.shape == (1, 2, 3)
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(rgb[0, 1], [10, 20, 30])


def test_load_rgb_image_converts_grayscale_to_three_channels(tmp_path: Path) -> None:
    grayscale = np.array(
        [
            [0, 64, 255],
            [10, 20, 30],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "image.png"
    Image.fromarray(grayscale, mode="L").save(image_path)

    rgb = load_rgb_image(image_path)

    assert rgb.shape == (2, 3, 3)
    np.testing.assert_array_equal(rgb[..., 0], grayscale)
    np.testing.assert_array_equal(rgb[..., 1], grayscale)
    np.testing.assert_array_equal(rgb[..., 2], grayscale)


def test_load_rgb_image_reports_invalid_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(ValueError, match="Could not load image"):
        load_rgb_image(missing_path)
