import albumentations as A
import cv2

from ultrasound_dg.configs.config_schemas import AugmentationConfig


def create_train_augmenter(config: AugmentationConfig, seed: int) -> A.Compose | None:
    """Create train-only geometric augmentation."""
    if not config.enabled:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=config.horizontal_flip_probability),
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
                        scale=(config.scale_min, config.scale_max),
                        translate_percent={
                            "x": (-config.translation_limit, config.translation_limit),
                            "y": (-config.translation_limit, config.translation_limit),
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
    )
