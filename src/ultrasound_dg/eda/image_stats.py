from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ultrasound_dg.data.dataset import load_image


def compute_image_stats(
    manifest: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    rows = []

    for _, row in manifest.iterrows():
        image_path = project_root / Path(row["image_path"])

        with Image.open(image_path) as image:
            image_width, image_height = image.size
            image_mode = image.mode
            image_bands = image.getbands()

            channels = len(image_bands)
            has_alpha = "A" in image_bands

            rgb = np.asarray(image.convert("RGB"))

        grayscale_image = load_image(image_path)

        # RGB mode does not necessarily mean that an ultrasound image contains  meaningful color information, as grayscale images can also be stored
        # using three identical RGB channels.

        # For each pixel, compute the difference between the largest and smallest  RGB channel value. Pixels with a channel difference > 5 are treated as
        # meaningfully colored, while smaller differences are ignored as possible encoding/compression noise.

        channel_spread = rgb.max(axis=-1).astype(np.int16) - rgb.min(axis=-1).astype(
            np.int16
        )
        # channel_difference_pixel_fraction represents the fraction of all pixels in the image
        # that contain noticeable color information.
        channel_difference_pixel_fraction = float((channel_spread > 5).mean())

        rgb_float = rgb.astype(np.float32)

        max_channel = rgb_float.max(axis=-1)
        min_channel = rgb_float.min(axis=-1)

        chroma = max_channel - min_channel

        # measure color strength relative to pixel brightness. The same RGB channel difference can be weak in a bright pixel
        # but strong in a dark one, so chroma is normalized by max_channel

        saturation = np.divide(
            chroma, max_channel, out=np.zeros_like(chroma), where=max_channel > 0
        )

        # Keep only clearly saturated and sufficiently bright pixels.
        # This helps distinguish real color content (e.g. Doppler)
        # from weak tint, compression noise, or very dark pixels.

        strong_color_mask = (saturation > 0.25) & (max_channel > 40)
        # max_channel > 40 prevents from recognizing very dark pixels as color, just because mathematically they have high saturation.
        strong_channel_difference_pixel_fraction = float(strong_color_mask.mean())

        rows.append(
            {
                "sample_id": row["sample_id"],
                "source_domain": row["source_domain"],
                "height": image_height,
                "width": image_width,
                "aspect_ratio": image_width / image_height,
                "image_mode": image_mode,
                "channels": channels,
                "has_alpha": has_alpha,
                "channel_difference_pixel_fraction": channel_difference_pixel_fraction,
                "strong_channel_difference_pixel_fraction": strong_channel_difference_pixel_fraction,
                "mean_intensity": float(grayscale_image.mean()),
                "std_intensity": float(grayscale_image.std()),
                "p01_intensity": float(np.percentile(grayscale_image, 1)),
                "p99_intensity": float(np.percentile(grayscale_image, 99)),
            }
        )
    stats = pd.DataFrame(rows)

    stats["dynamic_range"] = stats["p99_intensity"] - stats["p01_intensity"]

    return stats


def image_stats_summary(
    image_stats: pd.DataFrame,
) -> pd.DataFrame:
    summary = image_stats.groupby("source_domain").agg(
        image_count=("sample_id", "size"),
        # Image geometry
        min_height=("height", "min"),
        median_height=("height", "median"),
        max_height=("height", "max"),
        min_width=("width", "min"),
        median_width=("width", "median"),
        max_width=("width", "max"),
        min_aspect_ratio=("aspect_ratio", "min"),
        median_aspect_ratio=("aspect_ratio", "median"),
        max_aspect_ratio=("aspect_ratio", "max"),
        # Brightness across images
        mean_brightness=("mean_intensity", "mean"),
        median_brightness=("mean_intensity", "median"),
        brightness_q25=(
            "mean_intensity",
            lambda values: values.quantile(0.25),
        ),
        brightness_q75=(
            "mean_intensity",
            lambda values: values.quantile(0.75),
        ),
        # Contrast across images
        mean_contrast=("std_intensity", "mean"),
        median_contrast=("std_intensity", "median"),
        contrast_q25=(
            "std_intensity",
            lambda values: values.quantile(0.25),
        ),
        contrast_q75=(
            "std_intensity",
            lambda values: values.quantile(0.75),
        ),
        # Robust within-image intensity range
        mean_dynamic_range=("dynamic_range", "mean"),
        median_dynamic_range=("dynamic_range", "median"),
        dynamic_range_q25=(
            "dynamic_range",
            lambda values: values.quantile(0.25),
        ),
        dynamic_range_q75=(
            "dynamic_range",
            lambda values: values.quantile(0.75),
        ),
        # Color content
        mean_channel_difference_pixel_fraction=(
            "channel_difference_pixel_fraction",
            "mean",
        ),
        median_channel_difference_pixel_fraction=(
            "channel_difference_pixel_fraction",
            "median",
        ),
        mean_strong_channel_difference_pixel_fraction=(
            "strong_channel_difference_pixel_fraction",
            "mean",
        ),
        median_strong_channel_difference_pixel_fraction=(
            "strong_channel_difference_pixel_fraction",
            "median",
        ),
        strong_colored_image_ratio=(
            "strong_channel_difference_pixel_fraction",
            lambda values: (values > 0.01).mean(),
        ),
        # Storage properties
        alpha_image_ratio=("has_alpha", "mean"),
        unique_image_mode_count=("image_mode", "nunique"),
    )

    unique_resolutions = (
        image_stats[["source_domain", "height", "width"]]
        .drop_duplicates()
        .groupby("source_domain")
        .size()
        .rename("unique_resolution_count")
    )

    summary = summary.join(unique_resolutions)

    return summary.reset_index()


def doppler_stats_summary(
    image_stats: pd.DataFrame,
    manifest: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    doppler_stats = image_stats.merge(
        manifest[["sample_id", "doppler"]],
        on="sample_id",
        how="left",
    )

    doppler_stats = doppler_stats.dropna(subset=["doppler"]).copy()

    doppler_stats["flagged_by_color_heuristic"] = (
        doppler_stats["strong_channel_difference_pixel_fraction"] > threshold
    )

    summary = doppler_stats.groupby(
        ["source_domain", "doppler"],
        observed=True,
    ).agg(
        image_count=("sample_id", "size"),
        mean_strong_color_fraction=(
            "strong_channel_difference_pixel_fraction",
            "mean",
        ),
        median_strong_color_fraction=(
            "strong_channel_difference_pixel_fraction",
            "median",
        ),
        max_strong_color_fraction=(
            "strong_channel_difference_pixel_fraction",
            "max",
        ),
        flagged_image_count=(
            "flagged_by_color_heuristic",
            "sum",
        ),
        not_flagged_image_count=(
            "flagged_by_color_heuristic",
            lambda values: (~values).sum(),
        ),
        flagged_image_ratio=(
            "flagged_by_color_heuristic",
            "mean",
        ),
    )

    return summary.reset_index()
