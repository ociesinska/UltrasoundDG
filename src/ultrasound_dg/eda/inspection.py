from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.eda.visualization import DOMAIN_LABELS, DOMAIN_ORDER


def _format_value(
    value,
    precision: int = 2,
) -> str:
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (float, np.floating)):
        return f"{value:.{precision}f}"

    return str(value)


def _format_fraction(value) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.2%}"


def _load_image_and_mask(
    row: pd.Series,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
) -> tuple[np.ndarray, np.ndarray]:
    image_path = project_root / Path(row["image_path"])

    with Image.open(image_path) as pil_image:
        image = np.array(pil_image.convert("RGB"))

    if pd.isna(row["mask_path"]):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask_path = project_root / Path(row["mask_path"])
        source_domain = row["source_domain"]
        mask = adapters[source_domain].decode_mask(mask_path)

    mask_values = np.unique(mask)

    if not np.isin(mask_values, [0, 1]).all():
        raise ValueError(
            f"Decoded mask for {row['sample_id']} is not binary: values={mask_values}"
        )

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image-mask shape mismatch for "
            f"{row['sample_id']}: "
            f"image={image.shape[:2]}, "
            f"mask={mask.shape}"
        )

    return image, mask


def _draw_mask_overlay(
    ax,
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    ax.imshow(image)

    if mask.any():
        visible_mask = np.ma.masked_where(mask == 0, mask)

        ax.imshow(
            visible_mask,
            cmap="autumn",
            alpha=0.4,
            vmin=0,
            vmax=1,
        )

        ax.contour(
            mask,
            levels=[0.5],
            colors=["lime"],
            linewidths=1,
        )

    ax.axis("off")


def prepare_inspection_table(
    manifest: pd.DataFrame,
    image_stats: pd.DataFrame,
    mask_stats: pd.DataFrame,
) -> pd.DataFrame:
    mask_metrics = mask_stats[
        [
            "sample_id",
            "lesion_fraction",
            "bbox_fraction",
            "component_count",
        ]
    ]

    inspection_table = manifest.merge(
        image_stats,
        on=["sample_id", "source_domain"],
        how="left",
        validate="one_to_one",
    ).merge(
        mask_metrics,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    return inspection_table


def select_extreme_per_domain(
    data: pd.DataFrame,
    column: str,
    samples_per_domain: int,
    largest: bool,
) -> pd.DataFrame:
    domain_selections: list[pd.DataFrame] = []

    for _, domain_data in data.groupby(
        "source_domain",
        sort=False,
        observed=True,
    ):
        if largest:
            selected = domain_data.nlargest(samples_per_domain, column)
        else:
            selected = domain_data.nsmallest(samples_per_domain, column)

        domain_selections.append(selected)

    if not domain_selections:
        return data.iloc[0:0].copy()

    return pd.concat(domain_selections, ignore_index=True)


def select_inspection_cases(
    inspection_table: pd.DataFrame,
    samples_per_domain: int = 5,
) -> pd.DataFrame:
    selections = []

    largest_lesions = select_extreme_per_domain(
        data=inspection_table,
        column="lesion_fraction",
        samples_per_domain=samples_per_domain,
        largest=True,
    ).assign(inspection_reason="largest lesion")

    lesion_cases = inspection_table[inspection_table["lesion_fraction"] > 0]

    smallest_lesions = select_extreme_per_domain(
        data=lesion_cases,
        column="lesion_fraction",
        samples_per_domain=samples_per_domain,
        largest=False,
    ).assign(inspection_reason="smallest lesion")

    multi_component_cases = inspection_table[inspection_table["component_count"] > 1]

    multi_component = select_extreme_per_domain(
        data=multi_component_cases,
        column="component_count",
        samples_per_domain=samples_per_domain,
        largest=True,
    ).assign(inspection_reason="multiple components")

    brightest = select_extreme_per_domain(
        data=inspection_table,
        column="mean_intensity",
        samples_per_domain=samples_per_domain,
        largest=True,
    ).assign(inspection_reason="highest brightness")

    darkest = select_extreme_per_domain(
        data=inspection_table,
        column="mean_intensity",
        samples_per_domain=samples_per_domain,
        largest=False,
    ).assign(inspection_reason="lowest brightness")

    selections.extend(
        [
            largest_lesions,
            smallest_lesions,
            multi_component,
            brightest,
            darkest,
        ]
    )

    return pd.concat(selections, ignore_index=True).drop_duplicates(
        subset=["sample_id", "inspection_reason"]
    )


def display_inspection_cases(
    selection_cases: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
    output_dir: Path | None = None,
    show: bool = True,
) -> list[Figure]:
    figures: list[Figure] = []

    grouped_cases = selection_cases.groupby(
        ["inspection_reason", "source_domain"],
        sort=False,
        observed=True,
    )

    for (inspection_reason, source_domain), cases in grouped_cases:
        row_count = len(cases)

        figure, axes = plt.subplots(
            nrows=row_count,
            ncols=4,
            figsize=(18, 4 * row_count),
            squeeze=False,
            constrained_layout=True,
        )

        figure.suptitle(
            f"Inspection: {inspection_reason} — {source_domain}",
            fontsize=16,
        )

        for row_index, (_, row) in enumerate(cases.iterrows()):
            image, mask = _load_image_and_mask(
                row=row,
                project_root=project_root,
                adapters=adapters,
            )

            image_ax = axes[row_index, 0]
            mask_ax = axes[row_index, 1]
            overlay_ax = axes[row_index, 2]
            info_ax = axes[row_index, 3]

            # Original image
            image_ax.imshow(image)
            image_ax.set_title("Original image")
            image_ax.axis("off")

            # Binary mask
            mask_ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            mask_ax.set_title("Mask" if mask.any() else "Empty mask")
            mask_ax.axis("off")

            # Image-mask overlay
            _draw_mask_overlay(overlay_ax, image, mask)
            overlay_ax.set_title("Mask overlay")

            # Useful sample information
            info = "\n".join(
                [
                    f"Sample: {row['sample_id']}",
                    f"Domain: {row['source_domain']}",
                    f"Patient: {_format_value(row.get('patient_id'))}",
                    f"Diagnosis: {_format_value(row.get('diagnosis'))}",
                    f"Doppler: {_format_value(row.get('doppler'))}",
                    "",
                    f"Resolution: {row['width']:.0f} × {row['height']:.0f}",
                    f"Brightness: {_format_value(row.get('mean_intensity'))}",
                    f"Contrast: {_format_value(row.get('std_intensity'))}",
                    f"Dynamic range: {_format_value(row.get('dynamic_range'))}",
                    "",
                    (
                        "Lesion fraction: "
                        f"{_format_fraction(row.get('lesion_fraction'))}"
                    ),
                    (f"BBox fraction: {_format_fraction(row.get('bbox_fraction'))}"),
                    f"Components: {_format_value(row.get('component_count'))}",
                    "",
                    f"Reason: {inspection_reason}",
                ]
            )

            info_ax.axis("off")
            info_ax.text(
                0,
                1,
                info,
                ha="left",
                va="top",
                fontsize=10,
                family="monospace",
            )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"{inspection_reason}_{source_domain}".lower()
                .replace(" ", "_")
                .replace("/", "_")
            )

            figure.savefig(
                output_dir / f"{filename}.png",
                dpi=180,
                bbox_inches="tight",
            )

        figures.append(figure)

    if show:
        plt.show()

    return figures


def _brightness_extremes_figure(
    inspection_table: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
) -> Figure:
    figure, axes = plt.subplots(
        nrows=len(DOMAIN_ORDER),
        ncols=2,
        figsize=(10, 13),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, source_domain in enumerate(DOMAIN_ORDER):
        domain_data = inspection_table[
            inspection_table["source_domain"] == source_domain
        ].dropna(subset=["mean_intensity"])

        if domain_data.empty:
            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")
            continue

        extremes = [
            ("darkest", domain_data.nsmallest(1, "mean_intensity").iloc[0]),
            ("brightest", domain_data.nlargest(1, "mean_intensity").iloc[0]),
        ]

        for column_index, (extreme_name, sample) in enumerate(extremes):
            image, _ = _load_image_and_mask(
                row=sample,
                project_root=project_root,
                adapters=adapters,
            )
            ax = axes[row_index, column_index]
            ax.imshow(image)
            ax.set_title(
                f"{DOMAIN_LABELS[source_domain]} — {extreme_name}\n"
                f"mean intensity = {sample['mean_intensity']:.2f}",
                fontsize=10,
            )
            ax.axis("off")

    figure.suptitle(
        "Representative brightness extremes across domains",
        fontsize=16,
    )
    return figure


def _lesion_size_extremes_figure(
    inspection_table: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
) -> Figure:
    lesion_cases = inspection_table[inspection_table["lesion_fraction"] > 0]

    figure, axes = plt.subplots(
        nrows=len(DOMAIN_ORDER),
        ncols=2,
        figsize=(10, 13),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, source_domain in enumerate(DOMAIN_ORDER):
        domain_data = lesion_cases[
            lesion_cases["source_domain"] == source_domain
        ].dropna(subset=["lesion_fraction"])

        if domain_data.empty:
            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")
            continue

        extremes = [
            ("smallest", domain_data.nsmallest(1, "lesion_fraction").iloc[0]),
            ("largest", domain_data.nlargest(1, "lesion_fraction").iloc[0]),
        ]

        for column_index, (extreme_name, sample) in enumerate(extremes):
            image, mask = _load_image_and_mask(
                row=sample,
                project_root=project_root,
                adapters=adapters,
            )
            ax = axes[row_index, column_index]
            _draw_mask_overlay(ax, image, mask)
            ax.set_title(
                f"{DOMAIN_LABELS[source_domain]} — {extreme_name}\n"
                f"lesion fraction = {sample['lesion_fraction']:.2%}",
                fontsize=10,
            )

    figure.suptitle(
        "Representative lesion-size extremes across domains",
        fontsize=16,
    )
    return figure


def _multi_component_figure(
    inspection_table: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
) -> Figure:
    source_domains = ["busi", "bus_uclm"]
    figure, axes = plt.subplots(
        nrows=len(source_domains),
        ncols=2,
        figsize=(10, 8),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, source_domain in enumerate(source_domains):
        domain_data = inspection_table[
            (inspection_table["source_domain"] == source_domain)
            & (inspection_table["component_count"] > 1)
        ].dropna(subset=["component_count"])

        domain_data = domain_data.assign(
            component_spread=(
                domain_data["bbox_fraction"] / domain_data["lesion_fraction"]
            )
        )
        highest_count = domain_data.nlargest(
            1,
            ["component_count", "component_spread"],
        )
        remaining = domain_data.drop(index=highest_count.index)
        widest_spread = remaining.nlargest(1, "component_spread")
        selected = pd.concat([highest_count, widest_spread])

        for column_index, ax in enumerate(axes[row_index]):
            if column_index >= len(selected):
                ax.axis("off")
                continue

            sample = selected.iloc[column_index]
            image, mask = _load_image_and_mask(
                row=sample,
                project_root=project_root,
                adapters=adapters,
            )
            _draw_mask_overlay(ax, image, mask)
            ax.set_title(
                f"{DOMAIN_LABELS[source_domain]} — {sample['sample_id']}\n"
                f"components = {sample['component_count']:.0f}",
                fontsize=10,
            )

    figure.suptitle(
        "Representative multi-component masks",
        fontsize=16,
    )
    return figure


def save_qualitative_report_figures(
    inspection_table: pd.DataFrame,
    project_root: Path,
    adapters: Mapping[str, DatasetAdapter],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "brightness-extremes.png": _brightness_extremes_figure(
            inspection_table=inspection_table,
            project_root=project_root,
            adapters=adapters,
        ),
        "lesion-size-extremes.png": _lesion_size_extremes_figure(
            inspection_table=inspection_table,
            project_root=project_root,
            adapters=adapters,
        ),
        "multi-component-masks.png": _multi_component_figure(
            inspection_table=inspection_table,
            project_root=project_root,
            adapters=adapters,
        ),
    }

    output_paths: list[Path] = []

    for filename, figure in figures.items():
        output_path = output_dir / filename
        figure.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths
