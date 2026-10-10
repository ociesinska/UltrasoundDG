import numpy as np

from ultrasound_dg.configs.config_schemas import AugmentationConfig
from ultrasound_dg.data.augmentations import create_train_augmenter


def test_train_augmenter_samples_flip_and_geometry_independently() -> None:
    """Exercise all combinations of independently sampled augmentation groups."""
    augmenter = create_train_augmenter(
        config=AugmentationConfig(enabled=True),
        seed=42,
        save_applied_params=True,
    )

    assert augmenter is not None

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    observed_combinations: set[tuple[bool, bool]] = set()

    for _ in range(100):
        augmented = augmenter(image=image, mask=mask)
        transform_names = {name for name, _ in augmented["applied_transforms"]}

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


def test_train_augmenter_is_reproducible_for_a_fixed_seed() -> None:
    """Produce the same applied-transform sequence from the same seed."""
    config = AugmentationConfig(enabled=True)
    first = create_train_augmenter(
        config=config,
        seed=42,
        save_applied_params=True,
    )
    second = create_train_augmenter(
        config=config,
        seed=42,
        save_applied_params=True,
    )

    assert first is not None
    assert second is not None

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    first_sequence = [
        tuple(name for name, _ in first(image=image, mask=mask)["applied_transforms"])
        for _ in range(20)
    ]
    second_sequence = [
        tuple(name for name, _ in second(image=image, mask=mask)["applied_transforms"])
        for _ in range(20)
    ]

    assert first_sequence == second_sequence
