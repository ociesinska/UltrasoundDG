import numpy as np

from ultrasound_dg.configs.config_schemas import (
    ExperimentConfig,
    GeometricAugmentationConfig,
    IntensityAugmentationConfig,
    NoAugmentationConfig,
)
from ultrasound_dg.data.augmentations import create_augmentation_pipeline


def test_geometric_pipeline_samples_flip_and_geometry_independently() -> None:
    """Exercise all combinations of independently sampled geometric groups."""
    pipeline = create_augmentation_pipeline(
        config=GeometricAugmentationConfig(
            version="geometric_v1",
        ),
        seed=42,
        save_applied_params=True,
    )

    assert pipeline is not None
    assert pipeline.before_grayscale is not None
    assert pipeline.after_grayscale is None

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    observed_combinations: set[tuple[bool, bool]] = set()

    for _ in range(100):
        _, _, applied_transforms = pipeline.apply_before_grayscale(
            image=image,
            mask=mask,
        )
        transform_names = {name for name, _ in applied_transforms}

        observed_combinations.add(
            (
                "HorizontalFlip" in transform_names,
                bool({"SafeRotate", "Affine"} & transform_names),
            )
        )

    assert observed_combinations == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


def test_geometric_pipeline_is_reproducible_for_a_fixed_seed() -> None:
    """Produce the same geometric transform sequence from the same seed."""
    config = GeometricAugmentationConfig(version="geometric_v1")
    first = create_augmentation_pipeline(
        config=config,
        seed=42,
        save_applied_params=True,
    )
    second = create_augmentation_pipeline(
        config=config,
        seed=42,
        save_applied_params=True,
    )

    assert first is not None
    assert second is not None

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    first_sequence = [
        tuple(
            name for name, _ in first.apply_before_grayscale(image=image, mask=mask)[2]
        )
        for _ in range(20)
    ]
    second_sequence = [
        tuple(
            name for name, _ in second.apply_before_grayscale(image=image, mask=mask)[2]
        )
        for _ in range(20)
    ]

    assert first_sequence == second_sequence


def test_intensity_pipeline_uses_only_the_after_grayscale_stage() -> None:
    """Keep intensity transforms separate from paired geometric transforms."""
    pipeline = create_augmentation_pipeline(
        config=IntensityAugmentationConfig(
            version="intensity_v1",
            transform_probability=1.0,
        ),
        seed=42,
        save_applied_params=True,
    )

    assert pipeline is not None
    assert pipeline.before_grayscale is None
    assert pipeline.after_grayscale is not None

    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    augmented, applied_transforms = pipeline.apply_after_grayscale(image=image)

    assert augmented.shape == image.shape
    assert augmented.dtype == np.float32
    assert 0.0 <= augmented.min() <= augmented.max() <= 1.0
    assert len(applied_transforms) == 1


def test_intensity_pipeline_cannot_modify_the_mask() -> None:
    """Leave masks unchanged because intensity transforms are image-only."""
    pipeline = create_augmentation_pipeline(
        config=IntensityAugmentationConfig(version="intensity_v1"),
        seed=42,
    )

    assert pipeline is not None

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    unchanged_image, unchanged_mask, applied_transforms = (
        pipeline.apply_before_grayscale(
            image=image,
            mask=mask,
        )
    )

    np.testing.assert_array_equal(unchanged_image, image)
    np.testing.assert_array_equal(unchanged_mask, mask)
    assert applied_transforms == []


def test_intensity_pipeline_is_reproducible_for_a_fixed_seed() -> None:
    """Produce the same intensity outputs and transform sequence from one seed."""
    config = IntensityAugmentationConfig(version="intensity_v1")
    first = create_augmentation_pipeline(
        config=config,
        seed=42,
        save_applied_params=True,
    )
    second = create_augmentation_pipeline(
        config=config,
        seed=42,
        save_applied_params=True,
    )

    assert first is not None
    assert second is not None

    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    for _ in range(20):
        first_image, first_transforms = first.apply_after_grayscale(image=image)
        second_image, second_transforms = second.apply_after_grayscale(image=image)

        np.testing.assert_array_equal(first_image, second_image)
        assert [name for name, _ in first_transforms] == [
            name for name, _ in second_transforms
        ]


def test_disabled_augmentation_does_not_create_a_pipeline() -> None:
    """Return no training pipeline when augmentation is disabled."""
    pipeline = create_augmentation_pipeline(
        config=NoAugmentationConfig(),
        seed=42,
    )

    assert pipeline is None


def test_legacy_disabled_geometric_config_maps_to_no_augmentation() -> None:
    """Keep checkpoints created with the previous augmentation schema readable."""
    config = ExperimentConfig.model_validate(
        {
            "name": "legacy",
            "development": {
                "seed": 42,
                "source_domains": ["source"],
                "ood_development_domain": ["ood"],
                "final_test_domain": ["test"],
                "source_validation_fraction": 0.2,
                "splitting": {
                    "source": {
                        "strategy": "sample",
                    }
                },
            },
            "preprocessing": {},
            "augmentation": {
                "enabled": False,
                "version": "geometric_v1",
                "horizontal_flip_probability": 0.5,
            },
            "training": {
                "train_batch_size": 1,
                "eval_batch_size": 1,
                "epochs": 1,
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "decision_threshold": 0.5,
            },
        }
    )

    assert isinstance(config.augmentation, NoAugmentationConfig)
