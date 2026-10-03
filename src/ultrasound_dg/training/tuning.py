import logging
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import optuna
import torch
from torch.utils.data import DataLoader

from ultrasound_dg.configs.config_schemas import (
    ExperimentConfig,
    TrainingConfig,
    TuningConfig,
)
from ultrasound_dg.data.dataset import (
    UltrasoundSegmentationDataset,
)
from ultrasound_dg.data.samplers import create_domain_balanced_sampler
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.training.checkpoint_selection import (
    CheckpointCandidate,
    eligible_checkpoint_candidates,
    select_checkpoint_candidate,
)
from ultrasound_dg.training.checkpoints import (
    read_checkpoint,
    save_checkpoint,
)
from ultrasound_dg.training.evaluation import (
    evaluate_loader,
    macro_average_domain_metric,
)
from ultrasound_dg.training.losses import BCEDiceLoss
from ultrasound_dg.training.optimizers import create_optimizer
from ultrasound_dg.training.reproducibility import set_random_seed
from ultrasound_dg.training.train import train_one_epoch

logger = logging.getLogger(__name__)


def run_baseline_tuning_trial(
    trial: optuna.Trial,
    base_experiment_config: ExperimentConfig,
    tuning_config: TuningConfig,
    train_dataset: UltrasoundSegmentationDataset,
    source_val_loader: DataLoader,
    source_val_domain_loaders: dict[str, DataLoader],
    trial_checkpoint_dir: Path,
    device: torch.device,
) -> float:
    """Train and evaluate one Optuna hyperparameter trial.

    Learning rate and weight decay are sampled from the configured categorical
    grid. After each epoch, the objective reports macro-averaged source-domain
    lesion Dice for pruning. The best epoch and per-domain metrics are stored
    as trial metadata, and the best macro Dice is returned for optimization.
    """

    learning_rate = trial.suggest_categorical(
        "learning_rate",
        tuning_config.learning_rates,
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        tuning_config.weight_decays,
    )
    base_training_config = base_experiment_config.training
    trial_training_config = TrainingConfig.model_validate(
        {
            **base_training_config.model_dump(),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": tuning_config.epochs,
        }
    )

    trial_experiment_config = base_experiment_config.model_copy(
        update={
            "name": (
                tuning_config.output_experiment_name
                or f"{base_experiment_config.name}_tuned"
            ),
            "training": trial_training_config,
        }
    )

    set_random_seed(trial_training_config.seed)

    train_generator = torch.Generator()
    train_generator.manual_seed(trial_training_config.seed)

    train_sampler = None
    shuffle_train = True

    if base_training_config.sampling_strategy == "domain_balanced":
        train_sampler = create_domain_balanced_sampler(
            train_manifest=train_dataset.manifest, generator=train_generator
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=trial_training_config.train_batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=trial_training_config.num_workers,
        generator=train_generator if train_sampler is None else None,
    )

    model = create_unet().to(device)
    loss_fn = BCEDiceLoss().to(device)  # TODO: move loss function to config
    optimizer = create_optimizer(model=model, config=trial_training_config)

    checkpoint_candidates: list[CheckpointCandidate] = []
    best_observed_macro_dice = float("-inf")

    with TemporaryDirectory(prefix=f"ultrasound_dg_trial_{trial.number}_") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for epoch in range(trial_training_config.epochs):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )

            source_val_metrics_by_domain = {
                domain: evaluate_loader(
                    model=model,
                    loader=loader,
                    loss_fn=loss_fn,
                    device=device,
                    threshold=trial_training_config.decision_threshold,
                )
                for domain, loader in source_val_domain_loaders.items()
            }

            source_val_metrics = evaluate_loader(
                model=model,
                loader=source_val_loader,
                loss_fn=loss_fn,
                device=device,
                threshold=trial_training_config.decision_threshold,
            )

            macro_source_val_lesion_dice = macro_average_domain_metric(
                source_val_metrics_by_domain,
                metric_name="lesion_dice",
            )

            worst_domain_dice = min(
                metrics["lesion_dice"]
                for metrics in source_val_metrics_by_domain.values()
            )

            trial.report(macro_source_val_lesion_dice, step=epoch)

            logger.info(
                "Trial %d | epoch=%d/%d | train_loss=%.4f | "
                "macro_lesion_dice=%.4f | worst_domain_dice=%.4f",
                trial.number,
                epoch + 1,
                trial_training_config.epochs,
                train_loss,
                macro_source_val_lesion_dice,
                worst_domain_dice,
            )

            checkpoint_metrics = {
                **source_val_metrics,
                "macro_source_lesion_dice": macro_source_val_lesion_dice,
            }

            for domain, metrics in source_val_metrics_by_domain.items():
                for metric_name in (
                    "lesion_dice",
                    "lesion_iou",
                    "lesion_recall",
                    "lesion_precision",
                ):
                    checkpoint_metrics[f"{domain}_{metric_name}"] = metrics[metric_name]

            best_observed_macro_dice = max(
                best_observed_macro_dice, macro_source_val_lesion_dice
            )
            minimum_candidate_dice = (
                best_observed_macro_dice
                - trial_training_config.checkpoint_selection_tolerance
            )

            if macro_source_val_lesion_dice >= minimum_candidate_dice:
                candidate_path = tmp_dir / f"epoch_{epoch + 1}.pt"

                save_checkpoint(
                    path=candidate_path,
                    model=model,
                    optimizer=None,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                    configs=trial_experiment_config.model_dump(mode="json"),
                )

                checkpoint_candidates.append(
                    CheckpointCandidate(
                        path=candidate_path,
                        epoch=epoch + 1,
                        macro_lesion_dice=(macro_source_val_lesion_dice),
                        normal_fp_fraction=source_val_metrics["normal_fp_fraction"],
                    )
                )

            retained_candidates = eligible_checkpoint_candidates(
                candidates=checkpoint_candidates,
                tolerance=trial_training_config.checkpoint_selection_tolerance,
            )

            retained_paths = {candidate.path for candidate in retained_candidates}

            for candidate in checkpoint_candidates:
                if candidate.path not in retained_paths:
                    candidate.path.unlink(missing_ok=True)

            checkpoint_candidates = retained_candidates

            if trial.should_prune():
                raise optuna.TrialPruned()

        selected_epoch_candidate = select_checkpoint_candidate(
            candidates=checkpoint_candidates,
            tolerance=trial_training_config.checkpoint_selection_tolerance,
        )

        trial_checkpoint_path = trial_checkpoint_dir / f"trial_{trial.number:03d}.pt"
        trial_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(selected_epoch_candidate.path, trial_checkpoint_path)
        selected_checkpoint = read_checkpoint(path=trial_checkpoint_path)
        selected_metrics = selected_checkpoint["metrics"]

    trial.set_user_attr(
        "selected_epoch",
        selected_epoch_candidate.epoch,
    )
    trial.set_user_attr(
        "selected_normal_fp_fraction",
        selected_epoch_candidate.normal_fp_fraction,
    )
    trial.set_user_attr(
        "max_observed_macro_lesion_dice",
        best_observed_macro_dice,
    )
    trial.set_user_attr(
        "checkpoint_path",
        str(trial_checkpoint_path.resolve()),
    )

    for domain in source_val_domain_loaders:
        for metric_name in (
            "lesion_dice",
            "lesion_iou",
            "lesion_recall",
            "lesion_precision",
        ):
            trial.set_user_attr(
                f"{domain}_{metric_name}",
                selected_metrics[f"{domain}_{metric_name}"],
            )

    return selected_epoch_candidate.macro_lesion_dice
