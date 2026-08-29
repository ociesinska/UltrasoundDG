from pathlib import Path

import matplotlib.pyplot as plt

from ultrasound_dg.data.adapters.breast_usg import BreastUSGAdapter
from ultrasound_dg.data.adapters.bus_bra import BusBraAdapter
from ultrasound_dg.data.adapters.bus_uclm import BusUclmAdapter
from ultrasound_dg.data.adapters.busi import BusiAdapter
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.eda.image_stats import (
    compute_image_stats,
    doppler_stats_summary,
    image_stats_summary,
)
from ultrasound_dg.eda.inspection import (
    display_inspection_cases,
    prepare_inspection_table,
    save_qualitative_report_figures,
    select_inspection_cases,
)
from ultrasound_dg.eda.mask_stats import (
    compute_mask_stats,
    mask_stats_summary,
)
from ultrasound_dg.eda.summary import manifest_summary
from ultrasound_dg.eda.visualization import (
    plot_brightness_and_contrast,
    plot_diagnosis_distribution,
    plot_images_per_patient,
    plot_lesion_fraction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"
EDA_OUTPUT = PROJECT_ROOT / "outputs" / "eda"
FIGURES_OUTPUT = EDA_OUTPUT / "figures"
REPORT_FIGURES_OUTPUT = PROJECT_ROOT / "reports" / "eda" / "figures"


def main() -> None:
    EDA_OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)

    adapters = {
        "bus_bra": BusBraAdapter(DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(DATA_ROOT / "BrEaST"),
    }

    # Manifest-level statistics
    manifest_stats = manifest_summary(manifest)

    # Image-level statistics
    image_stats = compute_image_stats(
        manifest=manifest,
        project_root=PROJECT_ROOT,
    )
    image_summary = image_stats_summary(image_stats)

    # Mask-level statistics
    mask_stats = compute_mask_stats(
        manifest=manifest,
        project_root=PROJECT_ROOT,
        adapters=adapters,
    )
    mask_summary = mask_stats_summary(mask_stats)

    manifest_stats.to_csv(
        EDA_OUTPUT / "manifest_summary.csv",
        index=False,
    )

    image_stats.to_csv(
        EDA_OUTPUT / "image_stats.csv",
        index=False,
    )

    image_summary.to_csv(
        EDA_OUTPUT / "image_stats_summary.csv",
        index=False,
    )

    mask_stats.to_csv(
        EDA_OUTPUT / "mask_stats.csv",
        index=False,
    )

    mask_summary.to_csv(
        EDA_OUTPUT / "mask_stats_summary.csv",
        index=False,
    )

    print("\nMANIFEST SUMMARY")
    print(manifest_stats.to_string(index=False))

    print("\nIMAGE SUMMARY")
    print(image_summary.to_string(index=False))

    print("\nMASK SUMMARY")
    print(mask_summary.to_string(index=False))

    # Check if strong-color heuristic in fact recognizes Doppler images

    doppler_summary = doppler_stats_summary(
        image_stats=image_stats,
        manifest=manifest,
    )

    doppler_summary.to_csv(
        EDA_OUTPUT / "doppler_stats_summary.csv",
        index=False,
    )

    figures = [
        plot_diagnosis_distribution(
            manifest,
            FIGURES_OUTPUT / "diagnosis_distribution.png",
        ),
        plot_images_per_patient(
            manifest,
            FIGURES_OUTPUT / "images_per_patient.png",
        ),
        plot_lesion_fraction(
            mask_stats,
            FIGURES_OUTPUT / "lesion_fraction.png",
        ),
        plot_brightness_and_contrast(
            image_stats,
            FIGURES_OUTPUT / "brightness_contrast.png",
        ),
    ]

    for figure in figures:
        plt.close(figure)

    inspection_table = prepare_inspection_table(
        manifest=manifest,
        image_stats=image_stats,
        mask_stats=mask_stats,
    )

    save_qualitative_report_figures(
        inspection_table=inspection_table,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        output_dir=REPORT_FIGURES_OUTPUT,
    )

    selection_cases = select_inspection_cases(
        inspection_table,
        samples_per_domain=5,
    )

    inspection_figures = display_inspection_cases(
        selection_cases=selection_cases,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        output_dir=EDA_OUTPUT / "manual_checks",
        show=False,
    )

    for figure in inspection_figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
