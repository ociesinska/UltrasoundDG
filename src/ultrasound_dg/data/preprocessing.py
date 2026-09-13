from dataclasses import dataclass

import numpy as np
from PIL import Image

from ultrasound_dg.configs.config_schemas import (
    SegmentationPreprocessingConfig,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return np.array(
        Image.fromarray(image).convert("L"),
        dtype=np.uint8,
    )


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_imagenet(image: np.ndarray) -> np.ndarray:
    return np.clip(image * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


@dataclass
class SegmentationPreprocessor:
    config: SegmentationPreprocessingConfig

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:

        image = to_grayscale(image)

        image, mask = self._resize_and_pad(image, mask)
        image = image.astype(np.float32) / 255.0
        image = np.repeat(image[..., None], repeats=3, axis=-1)
        image = normalize_imagenet(image)
        mask = mask.astype(np.float32)

        return {
            "image": image,
            "mask": mask,
        }

    def _resize_and_pad(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]

        scale = min(self.config.target_size / height, self.config.target_size / width)

        new_height = round(height * scale)
        new_width = round(width * scale)

        image = np.array(
            Image.fromarray(image).resize(
                (new_width, new_height), resample=Image.Resampling.BILINEAR
            )
        )

        mask = np.array(
            Image.fromarray(mask).resize(
                (new_width, new_height), resample=Image.Resampling.NEAREST
            )
        )

        pad_height = self.config.target_size - new_height
        pad_width = self.config.target_size - new_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

        image = np.pad(image, padding, mode="constant", constant_values=0)

        mask = np.pad(mask, padding, mode="constant", constant_values=0)

        return image, mask
