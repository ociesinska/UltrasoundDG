from functools import partial
from pathlib import Path

import optuna

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import (
    DevelopmentConfig,
    SegmentationPreprocessingConfig,
    TrainingConfig,
    TuningConfig,
)
from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.adapters.breast_usg import BreastUSGAdapter
from ultrasound_dg.data.adapters.bus_bra import BusBraAdapter
from ultrasound_dg.data.adapters.bus_uclm import BusUclmAdapter
from ultrasound_dg.data.adapters.busi import BusiAdapter
from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.domain_loaders import create_domain_loaders
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.training.device import resolve_device
from ultrasound_dg.training.tuning import run_baseline_tuning_trial

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"
CONFIG_ROOT = PROJECT_ROOT / "src" / "ultrasound_dg" / "configs"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

DEVELOPMENT_CONFIG_PATH = CONFIG_ROOT / "development" / "v1.yaml"
PREPROCESSING_CONFIG_PATH = CONFIG_ROOT / "preprocessing" / "v1.yaml"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training" / "baseline_v1.yaml"
TUNING_CONFIG_PATH = CONFIG_ROOT / "tuning" / "baseline_v1.yaml"

TUNING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tuning"


def main() -> None:

    base_training_config = load_config(
        TRAINING_CONFIG_PATH,
        TrainingConfig,
    )
    development_config = load_config(
        DEVELOPMENT_CONFIG_PATH,
        DevelopmentConfig,
    )
    preprocessing_config = load_config(
        PREPROCESSING_CONFIG_PATH,
        SegmentationPreprocessingConfig,
    )
    tuning_config = load_config(
        TUNING_CONFIG_PATH,
        TuningConfig,
    )

    manifest = load_manifest(MANIFEST_PATH)

    adapters: dict[str, DatasetAdapter] = {
        "bus_bra": BusBraAdapter(DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(DATA_ROOT / "BrEaST"),
    }

    protocol = create_development_protocol(
        manifest=manifest,
        config=development_config,
    )

    preprocessor = SegmentationPreprocessor(
        config=preprocessing_config,
    )

    train_dataset = UltrasoundSegmentationDataset(
        manifest=protocol["train"],
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
    )

    source_val_domain_loaders = create_domain_loaders(
        manifest=protocol["source_val"],
        domains=development_config.source_domains,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        batch_size=base_training_config.eval_batch_size,
        num_workers=base_training_config.num_workers,
    )

    device = resolve_device(base_training_config.device)

    TUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    search_space = {
        "learning_rate": tuning_config.learning_rates,
        "weight_decay": tuning_config.weight_decays,
    }

    sampler = optuna.samplers.GridSampler(  # TODO: move to config
        search_space=search_space,
        seed=base_training_config.seed,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tuning_config.startup_trials,  # first 5 trials are not pruned for optuna to get the reference point
        n_warmup_steps=tuning_config.warmup_epochs,  # for each trial
    )

    objective = partial(
        run_baseline_tuning_trial,
        training_config=base_training_config,
        tuning_config=tuning_config,
        train_dataset=train_dataset,
        source_val_domain_loaders=source_val_domain_loaders,
        device=device,
    )

    database_path = TUNING_OUTPUT_DIR / f"{tuning_config.study_name}.db"
    storage_url = f"sqlite:///{database_path.resolve().as_posix()}"

    study = optuna.create_study(
        study_name=tuning_config.study_name,
        storage=storage_url,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    number_of_combinations = len(tuning_config.learning_rates) * len(
        tuning_config.weight_decays
    )

    study.optimize(
        objective,
        n_trials=number_of_combinations,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    results_path = TUNING_OUTPUT_DIR / f"{tuning_config.study_name}.csv"

    study.trials_dataframe().to_csv(results_path, index=False)
    print("\nBest trial")
    print(f"  number: {study.best_trial.number}")
    print(f"  macro lesion Dice: {study.best_value:.4f}")
    print(f"  parameters: {study.best_params}")
    print(f"  metadata: {study.best_trial.user_attrs}")
    print(f"  results saved to: {results_path}")


if __name__ == "__main__":
    main()
