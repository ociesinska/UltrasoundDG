import logging
from pathlib import Path

from ultrasound_dg.data.adapters.breast_usg import BreastUSGAdapter
from ultrasound_dg.data.adapters.bus_bra import BusBraAdapter
from ultrasound_dg.data.adapters.bus_uclm import BusUclmAdapter
from ultrasound_dg.data.adapters.busi import BusiAdapter
from ultrasound_dg.data.prepare import (
    load_manifest,
    samples_to_manifest,
    validate_manifest,
)
from ultrasound_dg.data.validation import validate_samples
from ultrasound_dg.utils.logger import format_logger

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"


def main() -> None:
    format_logger()

    adapters = {
        "bus_bra": BusBraAdapter(DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(DATA_ROOT / "BrEaST"),
    }

    all_samples = []

    for dataset_name, adapter in adapters.items():
        logger.info("Preparing %s...", dataset_name.upper())

        samples = adapter.samples()
        logger.info("Samples: %d", len(samples))

        validate_samples(
            adapter=adapter,
            samples=samples,
        )

        logger.info("Validation for %s adapter passed.", dataset_name.upper())

        all_samples.extend(samples)

    logger.info("Total samples: %d", len(all_samples))

    manifest = samples_to_manifest(all_samples, PROJECT_ROOT)
    print(manifest.head())

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    logger.info("Manifest saved to %s", MANIFEST_PATH)

    saved_manifest = load_manifest(MANIFEST_PATH)
    validate_manifest(saved_manifest, PROJECT_ROOT)

    logger.info(
        "Manifest validation passed for %d samples.",
        len(saved_manifest),
    )


if __name__ == "__main__":
    main()
