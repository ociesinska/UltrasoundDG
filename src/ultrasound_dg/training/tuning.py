from statistics import fmean

import optuna
import torch
from torch.utils.data import DataLoader

from ultrasound_dg.configs.config_schemas import (
    TrainingConfig,
    TuningConfig,
)
from ultrasound_dg.data.dataset import (
    UltrasoundSegmentationDataset,
)
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.training.evaluation import evaluate_loader
from ultrasound_dg.training.losses import BCEDiceLoss
from ultrasound_dg.training.optimizers import create_optimizer
from ultrasound_dg.training.reproducibility import set_random_seed
from ultrasound_dg.training.train import train_one_epoch


def run_baseline_tuning_trial(
    trial: optuna.Trial,
    training_config: TrainingConfig,
    tuning_config: TuningConfig,
    train_dataset: UltrasoundSegmentationDataset,
    source_val_domain_loaders: dict[str, DataLoader],
    device: torch.device,
) -> float:

    learning_rate = trial.suggest_categorical(
        "learning_rate",
        tuning_config.learning_rates,
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        tuning_config.weight_decays,
    )

    trial_config = TrainingConfig.model_validate(
        {
            **training_config.model_dump(),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": tuning_config.epochs,
        }
    )

    set_random_seed(trial_config.seed)

    train_generator = torch.Generator()
    train_generator.manual_seed(trial_config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=trial_config.train_batch_size,
        shuffle=True,
        num_workers=trial_config.num_workers,
        generator=train_generator,
    )

    model = create_unet().to(device)
    loss_fn = BCEDiceLoss().to(device)  # TODO: move loss function to config
    optimizer = create_optimizer(model=model, config=trial_config)

    best_macro_source_val_lesion_dice = float("-inf")
    best_epoch = 0
    best_domain_metrics = None

    for epoch in range(trial_config.epochs):
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
                threshold=trial_config.decision_threshold,
            )
            for domain, loader in source_val_domain_loaders.items()
        }

        macro_source_val_lesion_dice = fmean(
            metrics["lesion_dice"] for metrics in source_val_metrics_by_domain.values()
        )

        worst_domain_dice = min(
            metrics["lesion_dice"] for metrics in source_val_metrics_by_domain.values()
        )

        if macro_source_val_lesion_dice > best_macro_source_val_lesion_dice:
            best_macro_source_val_lesion_dice = macro_source_val_lesion_dice
            best_epoch = epoch + 1
            best_domain_metrics = source_val_metrics_by_domain

        trial.report(macro_source_val_lesion_dice, step=epoch)

        print(
            f"Trial {trial.number} | "
            f"epoch={epoch + 1} | "
            f"train_loss={train_loss:.4f} | "
            f"macro_lesion_dice={macro_source_val_lesion_dice:.4f} | "
            f"worst_domain_dice={worst_domain_dice:.4f}"
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    if best_domain_metrics is None:
        raise RuntimeError("Trial produced no validation metrics.")

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr(
        "worst_source_lesion_dice",
        min(metrics["lesion_dice"] for metrics in best_domain_metrics.values()),
    )

    for domain, metrics in best_domain_metrics.items():
        trial.set_user_attr(
            f"{domain}_lesion_dice",
            metrics["lesion_dice"],
        )
        trial.set_user_attr(
            f"{domain}_lesion_iou",
            metrics["lesion_iou"],
        )
        trial.set_user_attr(
            f"{domain}_lesion_recall",
            metrics["lesion_recall"],
        )
        trial.set_user_attr(
            f"{domain}_lesion_precision",
            metrics["lesion_precision"],
        )

    return best_macro_source_val_lesion_dice
