from pathlib import Path

import numpy as np

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import SegmentationPreprocessingConfig
from ultrasound_dg.data.preprocessing import (
    SegmentationPreprocessor,
    to_grayscale,
)


def test_preprocessing_config_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocessing.yaml"
    config_path.write_text("version: test\ntarget_size: 128\n")

    config = load_config(config_path, SegmentationPreprocessingConfig)

    assert config.version == "test"
    assert config.target_size == 128


def test_to_grayscale_returns_two_dimensional_uint8_image() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    grayscale = to_grayscale(rgb)

    assert grayscale.shape == (2, 2)
    assert grayscale.dtype == np.uint8


def test_preprocessor_resizes_pads_and_preserves_binary_mask() -> None:
    image = np.full((2, 4, 3), fill_value=255, dtype=np.uint8)
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    preprocessor = SegmentationPreprocessor(
        SegmentationPreprocessingConfig(target_size=8)
    )

    processed = preprocessor(image=image, mask=mask)
    processed_image = processed["image"]
    processed_mask = processed["mask"]

    assert processed_image.shape == (8, 8)
    assert processed_mask.shape == (8, 8)
    assert processed_image.dtype == np.float32
    assert processed_mask.dtype == np.float32
    assert processed_image.min() == 0.0
    assert processed_image.max() == 1.0
    assert set(np.unique(processed_mask)) <= {0.0, 1.0}

    np.testing.assert_array_equal(processed_image[:2], 0.0)
    np.testing.assert_array_equal(processed_image[2:6], 1.0)
    np.testing.assert_array_equal(processed_image[6:], 0.0)
