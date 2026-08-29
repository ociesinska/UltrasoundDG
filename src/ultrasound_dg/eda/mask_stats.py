from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.dataset import load_image

# Count connected lesion regions using 8-connectivity: pixels touching horizontally, vertically, or diagonally
# are treated as part of the same component.
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def compute_mask_stats(
    manifest: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
) -> pd.DataFrame:
    rows = []

    for _, row in manifest.iterrows():
        source_domain = row["source_domain"]

        if pd.isna(row["mask_path"]):
            image_path = project_root / Path(row["image_path"])
            image = load_image(image_path)

            mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8,
            )

        else:
            mask_path = project_root / Path(row["mask_path"])

            mask = adapters[source_domain].decode_mask(mask_path)

        lesion_area_pixels = int(mask.sum())
        lesion_fraction = lesion_area_pixels / mask.size

        # Label each connected lesion region and count how many separate components are present in the binary mask.
        _, component_count = label(
            mask,
            structure=CONNECTIVITY_8,
        )

        if lesion_area_pixels > 0:
            ys, xs = np.where(mask)

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1

            bbox_area = bbox_width * bbox_height
            bbox_fraction = bbox_area / mask.size

        else:
            bbox_width = np.nan
            bbox_height = np.nan
            bbox_area = np.nan
            bbox_fraction = np.nan

        rows.append(
            {
                "sample_id": row["sample_id"],
                "source_domain": source_domain,
                "diagnosis": row["diagnosis"],
                "has_lesion": row["has_lesion"],
                "lesion_area_pixels": lesion_area_pixels,
                "lesion_fraction": lesion_fraction,
                "component_count": int(component_count),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": bbox_area,
                "bbox_fraction": bbox_fraction,
            }
        )

    return pd.DataFrame(rows)


def mask_stats_summary(
    mask_stats: pd.DataFrame,
) -> pd.DataFrame:
    lesion_stats = mask_stats[mask_stats["lesion_area_pixels"] > 0]

    summary = lesion_stats.groupby("source_domain").agg(
        lesion_sample_count=("sample_id", "size"),
        min_lesion_fraction=(
            "lesion_fraction",
            "min",
        ),
        lesion_fraction_q25=(
            "lesion_fraction",
            lambda values: values.quantile(0.25),
        ),
        median_lesion_fraction=(
            "lesion_fraction",
            "median",
        ),
        lesion_fraction_q75=(
            "lesion_fraction",
            lambda values: values.quantile(0.75),
        ),
        max_lesion_fraction=(
            "lesion_fraction",
            "max",
        ),
        mean_bbox_fraction=(
            "bbox_fraction",
            "mean",
        ),
        median_bbox_fraction=(
            "bbox_fraction",
            "median",
        ),
        mean_component_count=(
            "component_count",
            "mean",
        ),
        max_component_count=(
            "component_count",
            "max",
        ),
        multi_component_ratio=(
            "component_count",
            lambda values: (values > 1).mean(),
        ),
    )

    total_sample_count = (
        mask_stats.groupby("source_domain").size().rename("total_sample_count")
    )

    summary = summary.join(total_sample_count)

    return summary.reset_index()
