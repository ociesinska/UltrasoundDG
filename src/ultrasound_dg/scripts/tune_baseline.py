import argparse
import logging
from functools import partial
from pathlib import Path
from shutil import copy2

import mlflow
import optuna
import yaml
from torch.utils.data import DataLoader

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
    get_checkpoint_dir,
)
from ultrasound_dg.training.checkpoint_selection import (
    CheckpointCandidate,
    select_checkpoint_candidate,
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
    """Tune baseline hyperparameters and save the best experiment config."""
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

    source_val_dataset = UltrasoundSegmentationDataset(
        manifest=protocol["source_val"],
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
    )

    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=base_training_config.eval_batch_size,
        shuffle=False,
        num_workers=base_training_config.num_workers,
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

    trial_checkpoint_dir = TUNING_OUTPUT_ROOT / tuning_run_name / "trial_checkpoints"

    objective = partial(
        run_baseline_tuning_trial,
        base_experiment_config=base_experiment_config,
        tuning_config=tuning_config,
        train_dataset=train_dataset,
        source_val_loader=source_val_loader,
        source_val_domain_loaders=source_val_domain_loaders,
        trial_checkpoint_dir=trial_checkpoint_dir,
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

        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ]

        if not completed_trials:
            raise RuntimeError("Tuning study produced no completed trials.")

        results_path = TUNING_OUTPUT_ROOT / f"{tuning_run_name}.csv"

        trial_candidates = [
            CheckpointCandidate(
                path=Path(trial.user_attrs["checkpoint_path"]),
                epoch=trial.user_attrs["selected_epoch"],
                macro_lesion_dice=float(trial.value),
                normal_fp_fraction=trial.user_attrs["selected_normal_fp_fraction"],
            )
            for trial in completed_trials
        ]

        selected_trial_candidate = select_checkpoint_candidate(
            trial_candidates,
            tolerance=base_training_config.checkpoint_selection_tolerance,
        )

        trials_by_checkpoint_path = {
            Path(trial.user_attrs["checkpoint_path"]): trial
            for trial in completed_trials
        }

        selected_trial = trials_by_checkpoint_path[selected_trial_candidate.path]

        best_trial_dice = max(float(trial.value) for trial in completed_trials)

        study.trials_dataframe().to_csv(results_path, index=False)
        logger.info(
            "Best trial | number=%d | macro_lesion_dice=%.4f | "
            "parameters=%s | metadata=%s",
            selected_trial.number,
            selected_trial.value,
            selected_trial.params,
            selected_trial.user_attrs,
        )
        logger.info("Tuning results saved to %s", results_path)

        mlflow.log_param(
            "selected_trial_number",
            selected_trial.number,
        )

        mlflow.log_metric(
            "selected_trial_macro_lesion_dice",
            float(selected_trial.value),
        )
        mlflow.log_metric(
            "max_trial_macro_lesion_dice",
            best_trial_dice,
        )

        mlflow.log_metric(
            "selected_trial_normal_fp_fraction",
            selected_trial.user_attrs["selected_normal_fp_fraction"],
        )
        mlflow.log_dict(
            selected_trial.params,
            "best_parameters.json",
        )

        mlflow.log_dict(
            selected_trial.user_attrs,
            "best_trial_metadata.json",
        )

        mlflow.log_artifact(
            str(results_path),
            artifact_path="tuning",
        )

        best_training_config = TrainingConfig.model_validate(
            {
                **base_training_config.model_dump(),
                **selected_trial.params,
                "epochs": tuning_config.epochs,
            }
        )

        output_experiment_name = (
            tuning_config.output_experiment_name
            or f"{base_experiment_config.name}_tuned"
        )

        best_experiment_config = ExperimentConfig.model_validate(
            {
                **base_experiment_config.model_dump(mode="json"),
                "name": output_experiment_name,
                "training": best_training_config.model_dump(mode="json"),
            }
        )

        final_checkpoint_dir = get_checkpoint_dir(
            experiment_name=best_experiment_config.name, seed=best_training_config.seed
        )

        final_checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        final_checkpoint_path = final_checkpoint_dir / "best_source_val.pt"
        copy2(
            selected_trial_candidate.path,
            final_checkpoint_path,
        )

        best_config_path = TUNING_OUTPUT_ROOT / f"{output_experiment_name}.yaml"

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
