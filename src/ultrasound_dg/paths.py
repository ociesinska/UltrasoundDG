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
EVALUATION_OUTPUT_ROOT = OUTPUT_ROOT / "evaluation"

REPORT_ROOT = PROJECT_ROOT / "reports"


def get_checkpoint_dir(
    experiment_name: str,
    seed: int,
) -> Path:
    """Return the seed-specific checkpoint directory for an experiment."""
    return CHECKPOINT_ROOT / experiment_name / f"seed_{seed}"


def get_evaluation_result_path(
    experiment_name: str,
    seed: int,
) -> Path:
    """Return the evaluation-result path for an experiment and seed."""
    return EVALUATION_OUTPUT_ROOT / experiment_name / f"seed_{seed}.json"
