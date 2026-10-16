from dataclasses import dataclass
from typing import Any

import albumentations as A
import cv2
import numpy as np

from ultrasound_dg.configs.config_schemas import (
    AugmentationConfig,
    GeometricAugmentationConfig,
    IntensityAugmentationConfig,
    NoAugmentationConfig,
)

AppliedTransform = tuple[str, dict[str, Any]]


@dataclass
class AugmentationPipeline:
    """Store augmentations executed at different preprocessing stages."""

    before_grayscale: A.Compose | None = None
    after_grayscale: A.Compose | None = None

    def apply_before_grayscale(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[AppliedTransform]]:
        """Apply paired transforms before grayscale conversion."""
        if self.before_grayscale is None:
            return image, mask, []

        augmented = self.before_grayscale(
            image=image,
            mask=mask,
        )

        return (
            augmented["image"],
            augmented["mask"],
            augmented.get("applied_transforms", []),
        )

    def apply_after_grayscale(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[AppliedTransform]]:
        """Apply image-only transforms to a grayscale image in [0, 1]."""
        if self.after_grayscale is None:
            return image, []

        augmented = self.after_grayscale(image=image)

        return (
            augmented["image"],
            augmented.get("applied_transforms", []),
        )


def create_augmentation_pipeline(
    config: AugmentationConfig,
    seed: int,
    save_applied_params: bool = False,
) -> AugmentationPipeline | None:
    """Create the augmentation pipeline selected by the experiment config."""
    if isinstance(config, NoAugmentationConfig):
        return None

    if isinstance(config, GeometricAugmentationConfig):
        return AugmentationPipeline(
            before_grayscale=_create_geometric_transform(
                config=config,
                seed=seed,
                save_applied_params=save_applied_params,
            )
        )

    if isinstance(config, IntensityAugmentationConfig):
        return AugmentationPipeline(
            after_grayscale=_create_intensity_transform(
                config=config,
                seed=seed,
                save_applied_params=save_applied_params,
            )
        )

    raise TypeError(f"Unsupported augmentation config: {type(config).__name__}")


def _create_geometric_transform(
    config: GeometricAugmentationConfig,
    seed: int,
    save_applied_params: bool,
) -> A.Compose:
    """Create the paired geometric transform sequence."""
    augmenter = A.Compose(
        [
            A.HorizontalFlip(
                p=config.horizontal_flip_probability,
            ),
            A.OneOf(
                [
                    A.SafeRotate(
                        limit=(
                            -config.rotation_limit_degrees,
                            config.rotation_limit_degrees,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                    A.Affine(
                        scale=(
                            config.scale_min,
                            config.scale_max,
                        ),
                        translate_percent={
                            "x": (
                                -config.translation_limit,
                                config.translation_limit,
                            ),
                            "y": (
                                -config.translation_limit,
                                config.translation_limit,
                            ),
                        },
                        rotate=0,
                        shear=0,
                        keep_ratio=True,
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                ],
                p=config.transform_probability,
            ),
        ],
        seed=seed,
        save_applied_params=save_applied_params,
    )

    # Use different seeds so flip and geometric transforms are sampled independently.
    augmenter.transforms[0].set_random_seed(seed + 1)
    augmenter.transforms[1].set_random_seed(seed + 2)

    return augmenter


def _create_intensity_transform(
    config: IntensityAugmentationConfig,
    seed: int,
    save_applied_params: bool,
) -> A.Compose:
    """Create the image-only intensity transform sequence."""
    return A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(
                            -config.brightness_limit,
                            config.brightness_limit,
                        ),
                        contrast_limit=(0.0, 0.0),
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(0.0, 0.0),
                        contrast_limit=(
                            -config.contrast_limit,
                            config.contrast_limit,
                        ),
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(
                            round(config.gamma_min * 100),
                            round(config.gamma_max * 100),
                        ),
                        p=1.0,
                    ),
                    A.GaussNoise(
                        std_range=(0.0, config.noise_std_max),
                        mean_range=(0.0, 0.0),
                        per_channel=False,
                        p=1.0,
                    ),
                    A.GaussianBlur(
                        blur_limit=(
                            config.blur_kernel_size,
                            config.blur_kernel_size,
                        ),
                        sigma_limit=(0.0, config.blur_sigma_max),
                        p=1.0,
                    ),
                ],
                p=config.transform_probability,
            ),
        ],
        seed=seed,
        save_applied_params=save_applied_params,
    )
