from pathlib import Path

import torch
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
from ultrasound_dg.data.prepare import load_manifest
from ultrasound_dg.data.preprocessing import (
    SegmentationPreprocessor,
)
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.training.checkpoints import save_checkpoint
from ultrasound_dg.training.device import resolve_device
from ultrasound_dg.training.evaluation import evaluate_loader
from ultrasound_dg.training.losses import BCEDiceLoss
from ultrasound_dg.training.optimizers import create_optimizer
from ultrasound_dg.training.reproducibility import set_random_seed
from ultrasound_dg.training.train import train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"
CONFIG_ROOT = PROJECT_ROOT / "src" / "ultrasound_dg" / "configs"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEVELOPMENT_CONFIG_PATH = CONFIG_ROOT / "development" / "v1.yaml"
PREPROCESSING_CONFIG_PATH = CONFIG_ROOT / "preprocessing" / "v1.yaml"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training" / "baseline_v1.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "baseline_v1"


def main() -> None:

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

    training_config = load_config(TRAINING_CONFIG_PATH, TrainingConfig)
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

    device = resolve_device(training_config.device)

    model = create_unet().to(device)
    loss_fn = BCEDiceLoss().to(device)

    optimizer = create_optimizer(
        model=model,
        config=training_config,
    )

    best_source_val_dice = float("-inf")
    for epoch in range(training_config.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        train_metrics = evaluate_loader(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=training_config.decision_threshold,
        )

        source_val_metrics = evaluate_loader(
            model=model,
            loader=source_val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=training_config.decision_threshold,
        )

        checkpoint_configs = {
            "training": training_config.model_dump(),
            "development": development_config.model_dump(),
            "preprocessing": preprocessing_config.model_dump(),
        }

        save_checkpoint(
            path=CHECKPOINT_DIR / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=source_val_metrics,
            configs=checkpoint_configs,
        )

        if source_val_metrics["dice"] > best_source_val_dice:
            best_source_val_dice = source_val_metrics["dice"]

            save_checkpoint(
                path=CHECKPOINT_DIR / "best_source_val.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=source_val_metrics,
                configs=checkpoint_configs,
            )

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={source_val_metrics['loss']:.4f} | "
            f"val_dice={source_val_metrics['dice']:.4f}"
        )


if __name__ == "__main__":
    main()
