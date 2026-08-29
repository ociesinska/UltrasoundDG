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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"


def main() -> None:
    adapters = {
        "bus_bra": BusBraAdapter(DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(DATA_ROOT / "BrEaST"),
    }

    all_samples = []

    for dataset_name, adapter in adapters.items():
        print(f"\nPreparing {dataset_name.upper()}...")

        samples = adapter.samples()
        print(f"Samples: {len(samples)}")

        validate_samples(
            adapter=adapter,
            samples=samples,
        )

        print(f"Validation for {dataset_name.upper()} adapter passed.")

        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    manifest = samples_to_manifest(all_samples, PROJECT_ROOT)
    print(manifest.head())

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"\nManifest saved to {MANIFEST_PATH}")

    saved_manifest = load_manifest(MANIFEST_PATH)
    validate_manifest(saved_manifest, PROJECT_ROOT)

    print(f"Manifest validation passed for {len(saved_manifest)} samples.")


if __name__ == "__main__":
    main()
