from pathlib import Path
from typing import Hashable, Mapping

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.sample import UltrasoundSample


def validate_image_mask_pairing(
    images_by_key: Mapping[Hashable, Path], masks_by_key: Mapping[Hashable, Path]
) -> None:
    """Validate that every image has a corresponding mask and vice versa."""

    image_keys = set(images_by_key)
    mask_keys = set(masks_by_key)

    missing_masks = image_keys - mask_keys
    missing_images = mask_keys - image_keys

    if missing_masks:
        raise ValueError(f"Missing masks for samples: {sorted(missing_masks)[:10]}")

    if missing_images:
        raise ValueError(f"Missing images for masks: {sorted(missing_images)[:10]}")


def validate_samples(
    adapter: DatasetAdapter,
    samples: list[UltrasoundSample],
) -> None:
    """Run dataset-specific pixel-level validation."""

    for sample in samples:
        adapter.validate_sample(sample)
