from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DOMAIN_ORDER = [
    "bus_bra",
    "busi",
    "bus_uclm",
    "breast_usg",
]

DOMAIN_LABELS = {
    "bus_bra": "BUS-BRA",
    "busi": "Curated BUSI",
    "bus_uclm": "BUS-UCLM",
    "breast_usg": "BrEaST",
}

DOMAIN_COLORS = {
    "bus_bra": "#4C78A8",
    "busi": "#F58518",
    "bus_uclm": "#54A24B",
    "breast_usg": "#B279A2",
}

DIAGNOSIS_ORDER = ["normal", "benign", "malignant"]

DIAGNOSIS_COLORS = {
    "normal": "#9CA3AF",
    "benign": "#4C78A8",
    "malignant": "#E45756",
}


def _style_axes(ax: Axes) -> None:
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(
    figure: Figure,
    output_path: Path | None,
) -> Figure:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    return figure


def _draw_violin_with_box(
    ax: Axes,
    data: pd.DataFrame,
    value_column: str,
) -> None:
    domains = [
        domain
        for domain in DOMAIN_ORDER
        if not data.loc[
            data["source_domain"] == domain,
            value_column,
        ]
        .dropna()
        .empty
    ]

    positions = [DOMAIN_ORDER.index(domain) + 1 for domain in domains]

    values = [
        data.loc[
            data["source_domain"] == domain,
            value_column,
        ]
        .dropna()
        .to_numpy()
        for domain in domains
    ]

    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, domain in zip(violins["bodies"], domains, strict=True):
        body.set_facecolor(DOMAIN_COLORS[domain])
        body.set_edgecolor("black")
        body.set_alpha(0.65)

    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
    )

    for box in boxplot["boxes"]:
        box.set_facecolor("white")
        box.set_alpha(0.8)

    ax.set_xticks(range(1, len(DOMAIN_ORDER) + 1))
    ax.set_xticklabels([DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])

    _style_axes(ax)


def plot_diagnosis_distribution(
    manifest: pd.DataFrame,
    output_path: Path | None = None,
) -> Figure:
    diagnosed_samples = manifest.dropna(subset=["source_domain", "diagnosis"])

    diagnosis_ratios = pd.crosstab(
        diagnosed_samples["source_domain"],
        diagnosed_samples["diagnosis"],
        normalize="index",
    )

    diagnosis_ratios = (
        diagnosis_ratios.reindex(
            index=DOMAIN_ORDER,
            columns=DIAGNOSIS_ORDER,
            fill_value=0,
        )
        .fillna(0)
        .mul(100)
    )

    sample_counts = (
        diagnosed_samples["source_domain"]
        .value_counts()
        .reindex(DOMAIN_ORDER, fill_value=0)
    )

    figure, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    x_positions = np.arange(len(DOMAIN_ORDER))
    bottoms = np.zeros(len(DOMAIN_ORDER))

    for diagnosis in DIAGNOSIS_ORDER:
        values = diagnosis_ratios[diagnosis].to_numpy()

        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=diagnosis.capitalize(),
            color=DIAGNOSIS_COLORS[diagnosis],
        )

        for x_position, value, bottom in zip(
            x_positions,
            values,
            bottoms,
            strict=True,
        ):
            if value >= 4:
                ax.text(
                    x_position,
                    bottom + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

        bottoms += values

    for x_position, sample_count in zip(
        x_positions,
        sample_counts,
        strict=True,
    ):
        ax.text(
            x_position,
            101.5,
            f"n={sample_count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Samples (%)")
    ax.set_title("Diagnosis distribution by source domain")
    ax.legend(
        title="Diagnosis",
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )

    _style_axes(ax)

    return _save_figure(figure, output_path)


def plot_images_per_patient(
    manifest: pd.DataFrame,
    output_path: Path | None = None,
) -> Figure:
    images_per_patient = (
        manifest.dropna(subset=["patient_id"])
        .groupby(
            ["source_domain", "patient_id"],
            observed=True,
        )
        .size()
        .rename("image_count")
        .reset_index()
    )

    domains_with_patient_ids = [
        domain
        for domain in DOMAIN_ORDER
        if domain in images_per_patient["source_domain"].unique()
    ]

    positions = [DOMAIN_ORDER.index(domain) + 1 for domain in domains_with_patient_ids]

    values_by_domain = [
        images_per_patient.loc[
            images_per_patient["source_domain"] == domain,
            "image_count",
        ].to_numpy()
        for domain in domains_with_patient_ids
    ]

    figure, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    boxplot = ax.boxplot(
        values_by_domain,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
    )

    for box, domain in zip(
        boxplot["boxes"],
        domains_with_patient_ids,
        strict=True,
    ):
        box.set_facecolor(DOMAIN_COLORS[domain])
        box.set_alpha(0.7)

    # Individual patients are added as slightly jittered points.
    random_generator = np.random.default_rng(seed=42)

    for position, values, domain in zip(
        positions,
        values_by_domain,
        domains_with_patient_ids,
        strict=True,
    ):
        jittered_positions = random_generator.normal(
            loc=position,
            scale=0.045,
            size=len(values),
        )

        ax.scatter(
            jittered_positions,
            values,
            s=10,
            alpha=0.2,
            color=DOMAIN_COLORS[domain],
            edgecolors="none",
        )

    busi_position = DOMAIN_ORDER.index("busi") + 1

    ax.text(
        busi_position,
        8,
        "Patient IDs\nunavailable",
        ha="center",
        va="center",
        color="dimgray",
    )

    ax.set_xticks(range(1, len(DOMAIN_ORDER) + 1))
    ax.set_xticklabels([DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])

    # The logarithmic scale keeps values 1–2 and 17–39 readable
    # on the same plot.
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "40"])
    ax.set_ylim(0.8, 50)

    ax.set_ylabel("Images per patient")
    ax.set_title("Patient structure by source domain")

    _style_axes(ax)

    return _save_figure(figure, output_path)


def plot_lesion_fraction(
    mask_stats: pd.DataFrame,
    output_path: Path | None = None,
) -> Figure:
    lesion_stats = mask_stats.loc[mask_stats["lesion_area_pixels"] > 0].copy()

    lesion_stats["lesion_percentage"] = lesion_stats["lesion_fraction"] * 100

    figure, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    _draw_violin_with_box(
        ax=ax,
        data=lesion_stats,
        value_column="lesion_percentage",
    )

    ax.set_ylabel("Lesion area (% of image)")
    ax.set_title("Lesion size by source domain\n(non-empty masks only)")

    return _save_figure(figure, output_path)


def plot_brightness_and_contrast(
    image_stats: pd.DataFrame,
    output_path: Path | None = None,
) -> Figure:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    _draw_violin_with_box(
        ax=axes[0],
        data=image_stats,
        value_column="mean_intensity",
    )
    axes[0].set_title("Brightness by source domain")
    axes[0].set_ylabel("Mean pixel intensity")

    _draw_violin_with_box(
        ax=axes[1],
        data=image_stats,
        value_column="std_intensity",
    )
    axes[1].set_title("Contrast by source domain")
    axes[1].set_ylabel("Pixel intensity standard deviation")

    figure.suptitle(
        "Image intensity distributions",
        fontsize=14,
    )

    return _save_figure(figure, output_path)
