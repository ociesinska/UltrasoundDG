import argparse
import logging
from functools import partial
from pathlib import Path

import mlflow
import optuna
import yaml

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import (
    ExperimentConfig,
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
from ultrasound_dg.paths import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    TUNING_OUTPUT_ROOT,
)
from ultrasound_dg.training.tuning import run_baseline_tuning_trial
from ultrasound_dg.utils.device import resolve_device
from ultrasound_dg.utils.logger import format_logger
from ultrasound_dg.utils.mlflow import (
    log_config,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def main() -> None:
    format_logger()

    parser = argparse.ArgumentParser(
        description="Tune a breast ultrasound segmentation model."
    )

    parser.add_argument(
        "--base_experiment_config",
        type=Path,
        required=True,
        help="Path to the base experiment configuration YAML file.",
    )

    parser.add_argument(
        "--tuning_experiment_config",
        type=Path,
        required=True,
        help="Path to the tuning experiment configuration YAML file.",
    )

    args = parser.parse_args()

    base_experiment_config = load_config(args.base_experiment_config, ExperimentConfig)
    development_config = base_experiment_config.development
    preprocessing_config = base_experiment_config.preprocessing
    tuning_config = load_config(args.tuning_experiment_config, TuningConfig)
    base_training_config = base_experiment_config.training

    tuning_run_name = f"{base_experiment_config.name}_{tuning_config.study_name}"

    manifest = load_manifest(MANIFEST_PATH)

    adapters: dict[str, DatasetAdapter] = {
        "bus_bra": BusBraAdapter(RAW_DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(RAW_DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(RAW_DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(RAW_DATA_ROOT / "BrEaST"),
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

    setup_mlflow(
        experiment_name=tuning_config.mlflow_experiment_name,
        set_experiment=True,
    )

    TUNING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
        base_training_config=base_training_config,
        tuning_config=tuning_config,
        train_dataset=train_dataset,
        source_val_domain_loaders=source_val_domain_loaders,
        device=device,
    )

    database_path = TUNING_OUTPUT_ROOT / f"{tuning_run_name}.db"
    storage_url = f"sqlite:///{database_path.resolve().as_posix()}"

    study = optuna.create_study(
        study_name=tuning_run_name,
        storage=storage_url,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    number_of_combinations = len(tuning_config.learning_rates) * len(
        tuning_config.weight_decays
    )

    with mlflow.start_run(run_name=f"{tuning_run_name}"):
        log_config(development_config, "development_config")
        log_config(preprocessing_config, "preprocessing_config")
        log_config(base_training_config, "base_training_config")
        log_config(tuning_config, "tuning_config")

        study.optimize(
            objective,
            n_trials=number_of_combinations,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        results_path = TUNING_OUTPUT_ROOT / f"{tuning_run_name}.csv"

        study.trials_dataframe().to_csv(results_path, index=False)
        logger.info(
            "Best trial | number=%d | macro_lesion_dice=%.4f | "
            "parameters=%s | metadata=%s",
            study.best_trial.number,
            study.best_value,
            study.best_params,
            study.best_trial.user_attrs,
        )
        logger.info("Tuning results saved to %s", results_path)

        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_metric("best_macro_lesion_dice", float(study.best_value))

        mlflow.log_dict(
            study.best_params,
            "best_parameters.json",
        )

        mlflow.log_dict(
            study.best_trial.user_attrs,
            "best_trial_metadata.json",
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="tuning",
        )

        best_training_config = TrainingConfig.model_validate(
            {
                **base_training_config.model_dump(),
                **study.best_params,
                "epochs": tuning_config.epochs,
            }
        )

        best_experiment_config = ExperimentConfig.model_validate(
            {
                **base_experiment_config.model_dump(mode="json"),
                "name": f"{base_experiment_config.name}_tuned",
                "training": best_training_config.model_dump(mode="json"),
            }
        )

        best_config_path = TUNING_OUTPUT_ROOT / f"{tuning_run_name}_best_train_cfg.yaml"

        with best_config_path.open("w") as file:
            yaml.safe_dump(
                best_experiment_config.model_dump(mode="json"),
                file,
                sort_keys=False,
            )

        logger.info("Best experiment config saved to %s", best_config_path)
        mlflow.log_artifact(str(best_config_path), artifact_path="tuning")


if __name__ == "__main__":
    main()
