import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import ExperimentConfig
from ultrasound_dg.data.adapters.breast_usg import BreastUSGAdapter
from ultrasound_dg.data.adapters.bus_bra import BusBraAdapter
from ultrasound_dg.data.adapters.bus_uclm import BusUclmAdapter
from ultrasound_dg.data.adapters.busi import BusiAdapter
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor
from ultrasound_dg.eda.image_stats import (
    compute_image_stats,
    doppler_stats_summary,
    image_stats_summary,
)
from ultrasound_dg.eda.inspection import (
    display_inspection_cases,
    plot_preprocessing_v1_examples,
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
from ultrasound_dg.paths import (
    EDA_OUTPUT_ROOT,
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    REPORT_ROOT,
)
from ultrasound_dg.utils.logger import format_logger

logger = logging.getLogger(__name__)

FIGURES_OUTPUT = EDA_OUTPUT_ROOT / "figures"
REPORT_FIGURES_OUTPUT = REPORT_ROOT / "eda" / "figures"
PREPROCESSING_REPORT_FIGURES_OUTPUT = REPORT_ROOT / "preprocessing" / "figures"


def main() -> None:
    """Generate dataset-level statistics, figures, and qualitative checks."""
    format_logger()

    parser = argparse.ArgumentParser(
        description=(
            "Run dataset-level EDA and generate preprocessing examples using "
            "the selected experiment configuration."
        )
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help=(
            "Path to the experiment YAML whose preprocessing configuration "
            "will be used for the preprocessing examples."
        ),
    )
    args = parser.parse_args()

    experiment_config = load_config(
        args.experiment_config,
        ExperimentConfig,
    )
    preprocessing_config = experiment_config.preprocessing

    logger.info("Running exploratory data analysis...")
    logger.info(
        "Using preprocessing from experiment %s",
        experiment_config.name,
    )

    EDA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)

    adapters = {
        "bus_bra": BusBraAdapter(RAW_DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(RAW_DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(RAW_DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(RAW_DATA_ROOT / "BrEaST"),
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
        EDA_OUTPUT_ROOT / "manifest_summary.csv",
        index=False,
    )

    image_stats.to_csv(
        EDA_OUTPUT_ROOT / "image_stats.csv",
        index=False,
    )

    image_summary.to_csv(
        EDA_OUTPUT_ROOT / "image_stats_summary.csv",
        index=False,
    )

    mask_stats.to_csv(
        EDA_OUTPUT_ROOT / "mask_stats.csv",
        index=False,
    )

    mask_summary.to_csv(
        EDA_OUTPUT_ROOT / "mask_stats_summary.csv",
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
        EDA_OUTPUT_ROOT / "doppler_stats_summary.csv",
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

    preprocessor = SegmentationPreprocessor(config=preprocessing_config)

    preprocessing_figure = plot_preprocessing_v1_examples(
        inspection_table=inspection_table,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        output_path=(
            PREPROCESSING_REPORT_FIGURES_OUTPUT
            / f"preprocessing-{preprocessing_config.version}-examples.png"
        ),
    )

    plt.close(preprocessing_figure)

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
        output_dir=EDA_OUTPUT_ROOT / "manual_checks",
        show=False,
    )

    for figure in inspection_figures:
        plt.close(figure)

    logger.info("EDA outputs saved to %s", EDA_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
