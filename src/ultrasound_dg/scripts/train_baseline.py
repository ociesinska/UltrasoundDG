import argparse
import logging
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import mlflow
import torch
from mlflow.models import infer_signature
from torch.utils.data import DataLoader

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import (
    ExperimentConfig,
)
from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.adapters.breast_usg import BreastUSGAdapter
from ultrasound_dg.data.adapters.bus_bra import BusBraAdapter
from ultrasound_dg.data.adapters.bus_uclm import BusUclmAdapter
from ultrasound_dg.data.adapters.busi import BusiAdapter
from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.domain_loaders import create_domain_loaders
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.data.preprocessing import (
    SegmentationPreprocessor,
)
from ultrasound_dg.data.samplers import create_domain_balanced_sampler
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.paths import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    get_checkpoint_dir,
)
from ultrasound_dg.training.checkpoint_selection import (
    CheckpointCandidate,
    eligible_checkpoint_candidates,
    select_checkpoint_candidate,
)
from ultrasound_dg.training.checkpoints import (
    load_checkpoint,
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
from ultrasound_dg.utils.device import resolve_device
from ultrasound_dg.utils.logger import format_logger
from ultrasound_dg.utils.mlflow import (
    log_config,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Train, checkpoint, and log a configured segmentation baseline."""
    format_logger()

    parser = argparse.ArgumentParser(
        description="Train a breast ultrasound segmentation baseline."
    )

    parser.add_argument(
        "--experiment_config",
        type=Path,
        required=True,
        help="Path to the complete experiment configuration YAML file.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional training-seed override.",
    )
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    adapters: dict[str, DatasetAdapter] = {
        "bus_bra": BusBraAdapter(RAW_DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(RAW_DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(RAW_DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(RAW_DATA_ROOT / "BrEaST"),
    }

    experiment_config = load_config(args.experiment_config, ExperimentConfig)
    if args.seed is not None:
        training_config = experiment_config.training.model_copy(
            update={"seed": args.seed}
        )

        experiment_config = experiment_config.model_copy(
            update={"training": training_config}
        )
    development_config = experiment_config.development
    preprocessing_config = experiment_config.preprocessing
    training_config = experiment_config.training

    checkpoint_dir = get_checkpoint_dir(
        experiment_name=experiment_config.name,
        seed=training_config.seed,
    )

    set_random_seed(training_config.seed)

    protocol = create_development_protocol(
        manifest=manifest,
        config=development_config,
    )

    preprocessor = SegmentationPreprocessor(config=preprocessing_config)

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

    train_generator = torch.Generator()
    train_generator.manual_seed(training_config.seed)

    train_sampler = None
    shuffle_train = True

    if training_config.sampling_strategy == "domain_balanced":
        train_sampler = create_domain_balanced_sampler(
            train_manifest=protocol["train"], generator=train_generator
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=training_config.num_workers,
        generator=train_generator if train_sampler is None else None,
    )

    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    source_val_domain_loaders = create_domain_loaders(
        manifest=protocol["source_val"],
        domains=development_config.source_domains,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        batch_size=training_config.eval_batch_size,
        num_workers=training_config.num_workers,
    )

    device = resolve_device(training_config.device)

    model = create_unet().to(device)
    loss_fn = BCEDiceLoss().to(device)

    setup_mlflow(
        experiment_name=experiment_config.name,
        set_experiment=True,
    )

    optimizer = create_optimizer(
        model=model,
        config=training_config,
    )

    run_name = args.run_name or f"{experiment_config.name}_seed_{training_config.seed}"

    with mlflow.start_run(run_name=run_name):
        log_config(development_config, "development_config")
        log_config(preprocessing_config, "preprocessing_config")
        log_config(training_config, "training_config")

        mlflow.log_params(
            {
                "architecture": "unet",
                "encoder": "resnet34",
                "encoder_weights": "imagenet",
                "optimizer": training_config.optimizer,
                "learning_rate": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
                "train_batch_size": training_config.train_batch_size,
                "epochs": training_config.epochs,
                "decision_threshold": training_config.decision_threshold,
                "seed": training_config.seed,
                "input_size": preprocessing_config.target_size,
                "device": str(device),
                "sampling_strategy": training_config.sampling_strategy,
                "checkpoint_selection": "lesion_dice_then_normal_fp",
                "checkpoint_selection_tolerance": training_config.checkpoint_selection_tolerance,
            }
        )

        checkpoint_candidates: list[CheckpointCandidate] = []
        best_observed_macro_dice = float("-inf")

        with TemporaryDirectory(
            prefix="ultrasound_dg_checkpoint_selection_"
        ) as candidate_dir:
            candidate_dir = Path(candidate_dir)

            for epoch in range(training_config.epochs):
                logger.info(
                    "Starting epoch %d/%d",
                    epoch + 1,
                    training_config.epochs,
                )

                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                )

                source_val_metrics = evaluate_loader(
                    model=model,
                    loader=source_val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    threshold=training_config.decision_threshold,
                )

                source_val_metrics_by_domain = {
                    domain: evaluate_loader(
                        model=model,
                        loader=loader,
                        loss_fn=loss_fn,
                        device=device,
                        threshold=training_config.decision_threshold,
                    )
                    for domain, loader in source_val_domain_loaders.items()
                }

                macro_source_val_lesion_dice = macro_average_domain_metric(
                    source_val_metrics_by_domain,
                    metric_name="lesion_dice",
                )

                checkpoint_configs = experiment_config.model_dump(mode="json")

                checkpoint_metrics = {
                    **source_val_metrics,
                    "macro_source_lesion_dice": macro_source_val_lesion_dice,
                }

                for domain, metrics in source_val_metrics_by_domain.items():
                    checkpoint_metrics[f"{domain}_lesion_dice"] = metrics["lesion_dice"]

                save_checkpoint(
                    path=checkpoint_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                    configs=checkpoint_configs,
                )

                best_observed_macro_dice = max(
                    best_observed_macro_dice, macro_source_val_lesion_dice
                )
                minimum_candidate_dice = (
                    best_observed_macro_dice
                    - training_config.checkpoint_selection_tolerance
                )

                if macro_source_val_lesion_dice >= minimum_candidate_dice:
                    candidate_path = candidate_dir / f"epoch_{epoch + 1}.pt"

                    save_checkpoint(
                        path=candidate_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        metrics=checkpoint_metrics,
                        configs=checkpoint_configs,
                    )

                    checkpoint_candidates.append(
                        CheckpointCandidate(
                            path=candidate_path,
                            epoch=epoch + 1,
                            macro_lesion_dice=macro_source_val_lesion_dice,
                            normal_fp_fraction=source_val_metrics["normal_fp_fraction"],
                        )
                    )

                retained_candidates = eligible_checkpoint_candidates(
                    checkpoint_candidates,
                    tolerance=training_config.checkpoint_selection_tolerance,
                )
                retained_paths = {candidate.path for candidate in retained_candidates}

                for candidate in checkpoint_candidates:
                    if candidate.path not in retained_paths:
                        candidate.path.unlink(missing_ok=True)

                checkpoint_candidates = retained_candidates

                domain_mlflow_metrics = {
                    f"source_val_{domain}_lesion_dice": float(metrics["lesion_dice"])
                    for domain, metrics in source_val_metrics_by_domain.items()
                }

                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "source_val_loss": float(source_val_metrics["loss"]),
                        "source_val_dice": float(source_val_metrics["dice"]),
                        "source_val_lesion_dice": float(
                            source_val_metrics["lesion_dice"]
                        ),
                        "source_val_lesion_recall": float(
                            source_val_metrics["lesion_recall"]
                        ),
                        "source_val_lesion_precision": float(
                            source_val_metrics["lesion_precision"]
                        ),
                        "macro_source_lesion_dice": float(macro_source_val_lesion_dice),
                        **domain_mlflow_metrics,
                    },
                    step=epoch + 1,
                )

            selected_candidate = select_checkpoint_candidate(
                checkpoint_candidates,
                tolerance=training_config.checkpoint_selection_tolerance,
            )
            best_checkpoint_path = checkpoint_dir / "best_source_val.pt"

            copy2(
                selected_candidate.path,
                best_checkpoint_path,
            )

        best_checkpoint = load_checkpoint(
            path=checkpoint_dir / "best_source_val.pt",
            model=model,
            optimizer=None,
            device=device,
        )

        mlflow.log_param("best_epoch", best_checkpoint["epoch"])
        mlflow.log_metrics(
            {
                "max_observed_macro_source_lesion_dice": (best_observed_macro_dice),
                "selected_macro_source_lesion_dice": float(
                    best_checkpoint["metrics"]["macro_source_lesion_dice"]
                ),
                "selected_source_normal_fp_fraction": float(
                    best_checkpoint["metrics"]["normal_fp_fraction"]
                ),
            }
        )

        model = model.to("cpu")
        model.eval()

        example_batch = next(iter(source_val_loader))
        input_example = example_batch["image"][:1].cpu()

        with torch.no_grad():
            output_example = model(input_example)

        signature = infer_signature(
            input_example.numpy(), output_example.detach().numpy()
        )

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="best_model",
            input_example=input_example,
            signature=signature,
            serialization_format="pt2",
            metadata={
                "checkpoint_epoch": best_checkpoint["epoch"],
                "expects_preprocessed_input": True,
                "output_type": "segmentation_logits",
            },
        )
        logger.info("MLflow model URI: %s", model_info.model_uri)


if __name__ == "__main__":
    main()
