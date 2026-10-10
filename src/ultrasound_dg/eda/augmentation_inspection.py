from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.preprocessing import denormalize_imagenet
from ultrasound_dg.eda.inspection import draw_mask_overlay


def select_augmentation_examples(
    train_manifest: pd.DataFrame,
    source_domains: Sequence[str],
    samples_per_group: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Select lesion and normal training examples from every source domain."""
    selections: list[pd.DataFrame] = []

    for domain_index, source_domain in enumerate(source_domains):
        domain_samples = train_manifest[
            train_manifest["source_domain"] == source_domain
        ]

        lesion_samples = domain_samples[domain_samples["has_lesion"]]

        if not lesion_samples.empty:
            selections.append(
                lesion_samples.sample(
                    n=min(samples_per_group, len(lesion_samples)),
                    random_state=seed + domain_index,
                )
            )

        normal_samples = domain_samples[~domain_samples["has_lesion"]]

        if not normal_samples.empty:
            selections.append(
                normal_samples.sample(
                    n=min(samples_per_group, len(normal_samples)),
                    random_state=seed + 100 + domain_index,
                )
            )

    if not selections:
        return train_manifest.iloc[0:0].copy()

    return pd.concat(selections, ignore_index=True)


def _sample_to_display_arrays(
    sample: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a processed dataset sample into display-ready arrays."""
    image = sample["image"].permute(1, 2, 0).numpy()
    image = denormalize_imagenet(image)

    mask = sample["mask"].squeeze(0).numpy()

    return image, mask


def format_applied_transforms(
    applied_transforms: list[tuple[str, dict]],
) -> str:
    """Format applied Albumentations parameters for a figure title."""
    if not applied_transforms:
        return "No transform applied"

    descriptions: list[str] = []

    for transform_name, params in applied_transforms:
        if transform_name == "HorizontalFlip":
            descriptions.append("horizontal flip")

        elif transform_name == "SafeRotate":
            angle = float(params["rotate"])

            descriptions.append(f"rotation: {angle:+.2f}°")

        elif transform_name == "Affine":
            scale = float(params["scale"]["x"])
            matrix = np.asarray(params["matrix"])

            height, width = params["shape"][:2]
            center_x = (width - 1) / 2
            center_y = (height - 1) / 2

            translation_x_pixels = matrix[0, 2] - (1 - scale) * center_x
            translation_y_pixels = matrix[1, 2] - (1 - scale) * center_y

            translation_x = translation_x_pixels / width
            translation_y = translation_y_pixels / height

            descriptions.append(
                f"scale: {scale:.3f}, "
                f"translation: "
                f"x={translation_x:+.2%}, "
                f"y={translation_y:+.2%}"
            )

        else:
            descriptions.append(transform_name)

    return "\n".join(descriptions)


def plot_augmentation_examples(
    reference_dataset: UltrasoundSegmentationDataset,
    augmented_dataset: UltrasoundSegmentationDataset,
    variants_per_sample: int,
    output_path: Path,
) -> Figure:
    """Compare each unaugmented sample with several stochastic variants."""
    if len(reference_dataset) != len(augmented_dataset):
        raise ValueError(
            "Reference and augmented datasets must contain the same samples."
        )

    if len(reference_dataset) == 0:
        raise ValueError("No samples were selected for augmentation inspection.")

    column_count = variants_per_sample + 1

    figure, axes = plt.subplots(
        nrows=len(reference_dataset),
        ncols=column_count,
        figsize=(4 * column_count, 4 * len(reference_dataset)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index in range(len(reference_dataset)):
        reference_sample = reference_dataset[row_index]
        reference_image, reference_mask = _sample_to_display_arrays(reference_sample)

        draw_mask_overlay(
            axes[row_index, 0],
            reference_image,
            reference_mask,
        )

        axes[row_index, 0].set_title(
            "Original processed input\n"
            f"{reference_sample['source_domain']} | "
            f"{reference_sample['sample_id']}\n"
            f"{reference_sample['diagnosis']}"
        )

        for variant_index in range(variants_per_sample):
            augmented_sample = augmented_dataset[row_index]
            augmented_image, augmented_mask = _sample_to_display_arrays(
                augmented_sample
            )
            applied_transforms = augmented_sample.get(
                "applied_transforms",
                [],
            )

            transform_description = format_applied_transforms(applied_transforms)

            draw_mask_overlay(
                axes[row_index, variant_index + 1],
                augmented_image,
                augmented_mask,
            )

            axes[row_index, variant_index + 1].set_title(
                f"Augmented variant {variant_index + 1}\n{transform_description}",
                fontsize=9,
            )

    figure.suptitle(
        "Geometric augmentation quality control",
        fontsize=16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    return figure
