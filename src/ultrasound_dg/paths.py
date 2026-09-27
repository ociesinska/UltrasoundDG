from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

CONFIG_ROOT = PACKAGE_ROOT / "configs"

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
MANIFEST_ROOT = DATA_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_ROOT / "all_samples.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
TUNING_OUTPUT_ROOT = OUTPUT_ROOT / "tuning"
EDA_OUTPUT_ROOT = OUTPUT_ROOT / "eda"

REPORT_ROOT = PROJECT_ROOT / "reports"


def get_checkpoint_dir(
    experiment_name: str,
    seed: int,
) -> Path:
    return CHECKPOINT_ROOT / experiment_name / f"seed_{seed}"
