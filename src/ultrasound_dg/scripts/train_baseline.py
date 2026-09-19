import argparse
import logging
from pathlib import Path

import mlflow
import torch
from mlflow.models import infer_signature
from torch.utils.data import DataLoader

from ultrasound_dg.configs.config_loader import load_config
from ultrasound_dg.configs.config_schemas import (
    DevelopmentConfig,
    SegmentationPreprocessingConfig,
    TrainingConfig,
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
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.models.unet import create_unet
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
from ultrasound_dg.utils.mlflow import (
    log_config,
    setup_mlflow,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"
CONFIG_ROOT = PROJECT_ROOT / "src" / "ultrasound_dg" / "configs"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEVELOPMENT_CONFIG_PATH = CONFIG_ROOT / "development" / "v1.yaml"
PREPROCESSING_CONFIG_PATH = CONFIG_ROOT / "preprocessing" / "v1.yaml"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training" / "baseline_v1.yaml"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train a breast ultrasound segmentation baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAINING_CONFIG_PATH,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    adapters: dict[str, DatasetAdapter] = {
        "bus_bra": BusBraAdapter(DATA_ROOT / "BUSBRA"),
        "busi": BusiAdapter(DATA_ROOT / "BUSI_Curated"),
        "bus_uclm": BusUclmAdapter(DATA_ROOT / "BUS-UCLM"),
        "breast_usg": BreastUSGAdapter(DATA_ROOT / "BrEaST"),
    }

    preprocessing_config = load_config(
        PREPROCESSING_CONFIG_PATH, SegmentationPreprocessingConfig
    )

    development_config = load_config(
        DEVELOPMENT_CONFIG_PATH,
        DevelopmentConfig,
    )

    training_config = load_config(args.config, TrainingConfig)

    checkpoint_dir = (
        CHECKPOINT_ROOT
        / training_config.mlflow_experiment_name
        / f"seed_{training_config.seed}"
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        generator=train_generator,
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
        experiment_name=training_config.mlflow_experiment_name,
        tracking_uri=MLFLOW_TRACKING_URI,
        set_experiment=True,
    )

    optimizer = create_optimizer(
        model=model,
        config=training_config,
    )

    with mlflow.start_run(run_name=f"{training_config.mlflow_experiment_name}"):
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
            }
        )

        best_macro_source_val_lesion_dice = float("-inf")
        for epoch in range(training_config.epochs):
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

            checkpoint_configs = {
                "training": training_config.model_dump(),
                "development": development_config.model_dump(),
                "preprocessing": preprocessing_config.model_dump(),
            }

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

            if macro_source_val_lesion_dice > best_macro_source_val_lesion_dice:
                best_macro_source_val_lesion_dice = macro_source_val_lesion_dice

                save_checkpoint(
                    path=checkpoint_dir / "best_source_val.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                    configs=checkpoint_configs,
                )

            logger.info(
                f"Epoch {epoch + 1} | "
                f"train_loss={train_loss:.4f} | "
                f"source_val_loss={source_val_metrics['loss']:.4f} | "
                f"source_val_dice={source_val_metrics['dice']:.4f} | "
                f"macro_source_lesion_dice="
                f"{macro_source_val_lesion_dice:.4f}"
            )

            domain_mlflow_metrics = {
                f"source_val_{domain}_lesion_dice": float(metrics["lesion_dice"])
                for domain, metrics in source_val_metrics_by_domain.items()
            }
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "source_val_loss": float(source_val_metrics["loss"]),
                    "source_val_dice": float(source_val_metrics["dice"]),
                    "source_val_lesion_dice": float(source_val_metrics["lesion_dice"]),
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

        best_checkpoint = load_checkpoint(
            path=checkpoint_dir / "best_source_val.pt",
            model=model,
            optimizer=None,
            device=device,
        )

        mlflow.log_param("best_epoch", best_checkpoint["epoch"])
        mlflow.log_metric(
            "best_macro_source_lesion_dice",
            float(best_checkpoint["metrics"]["macro_source_lesion_dice"]),
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
