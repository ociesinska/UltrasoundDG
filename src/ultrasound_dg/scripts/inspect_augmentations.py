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
from ultrasound_dg.data.augmentations import create_train_augmenter
from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.eda.augmentation_inspection import (
    plot_augmentation_examples,
    select_augmentation_examples,
)
from ultrasound_dg.paths import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
)
from ultrasound_dg.utils.logger import format_logger

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate a visual quality-control grid for training augmentations."""
    format_logger()

    parser = argparse.ArgumentParser(
        description="Visualize paired image-mask training augmentations."
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--variants-per-sample",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    experiment_config = load_config(
        args.experiment_config,
        ExperimentConfig,
    )

    if not experiment_config.augmentation.enabled:
        raise ValueError("The selected experiment does not enable augmentation.")

    manifest = load_manifest(MANIFEST_PATH)

    adapters = {
        "bus_bra": BusBraAdapter(RAW_DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(RAW_DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(RAW_DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(RAW_DATA_ROOT / "BrEaST"),
    }

    protocol = create_development_protocol(
        manifest=manifest,
        config=experiment_config.development,
    )

    selected_manifest = select_augmentation_examples(
        train_manifest=protocol["train"],
        source_domains=experiment_config.development.source_domains,
        samples_per_group=args.samples_per_group,
        seed=args.seed,
    )

    preprocessor = SegmentationPreprocessor(config=experiment_config.preprocessing)

    augmenter = create_train_augmenter(
        config=experiment_config.augmentation,
        seed=args.seed,
        save_applied_params=True,
    )

    reference_dataset = UltrasoundSegmentationDataset(
        manifest=selected_manifest,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        augmenter=None,
    )

    augmented_dataset = UltrasoundSegmentationDataset(
        manifest=selected_manifest,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        augmenter=augmenter,
    )

    figure = plot_augmentation_examples(
        reference_dataset=reference_dataset,
        augmented_dataset=augmented_dataset,
        variants_per_sample=args.variants_per_sample,
        output_path=args.output,
    )

    plt.close(figure)

    logger.info(f"Augmentation inspection saved to {args.output}")


if __name__ == "__main__":
    main()
