import argparse
import logging
from pathlib import Path
from statistics import fmean

from torch.utils.data import DataLoader

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
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.paths import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    get_evaluation_result_path,
)
from ultrasound_dg.training.checkpoints import (
    read_checkpoint,
    restore_checkpoint,
)
from ultrasound_dg.training.evaluation import evaluate_loader
from ultrasound_dg.training.losses import BCEDiceLoss
from ultrasound_dg.training.metrics import print_metrics
from ultrasound_dg.utils.device import resolve_device
from ultrasound_dg.utils.logger import format_logger
from ultrasound_dg.utils.serialization import save_json

logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate a checkpoint on source validation and OOD development data."""
    format_logger()

    parser = argparse.ArgumentParser(
        description="Evaluate a breast ultrasound segmentation model."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint.",
    )

    args = parser.parse_args()

    checkpoint = read_checkpoint(path=args.checkpoint, map_location="cpu")
    experiment_config = ExperimentConfig.model_validate(checkpoint["configs"])
    development_config = experiment_config.development
    preprocessing_config = experiment_config.preprocessing
    training_config = experiment_config.training

    device = resolve_device(training_config.device)
    model = create_unet().to(device)

    restore_checkpoint(checkpoint=checkpoint, model=model, optimizer=None)

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
    preprocessor = SegmentationPreprocessor(config=preprocessing_config)

    source_val_domain_loaders = create_domain_loaders(
        manifest=protocol["source_val"],
        domains=development_config.source_domains,
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
        batch_size=training_config.eval_batch_size,
        num_workers=training_config.num_workers,
    )

    source_val_dataset = UltrasoundSegmentationDataset(
        manifest=protocol["source_val"],
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
    )
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    ood_dev_dataset = UltrasoundSegmentationDataset(
        manifest=protocol["ood_dev"],
        project_root=PROJECT_ROOT,
        adapters=adapters,
        preprocessor=preprocessor,
    )
    ood_dev_loader = DataLoader(
        ood_dev_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    loss_fn = BCEDiceLoss().to(device)
    source_val_metrics = evaluate_loader(
        model=model,
        loader=source_val_loader,
        loss_fn=loss_fn,
        device=device,
        threshold=training_config.decision_threshold,
    )
    ood_metrics = evaluate_loader(
        model=model,
        loader=ood_dev_loader,
        loss_fn=loss_fn,
        device=device,
        threshold=training_config.decision_threshold,
    )

    source_domain_metrics = {
        domain: evaluate_loader(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            threshold=training_config.decision_threshold,
        )
        for domain, loader in source_val_domain_loaders.items()
    }

    macro_source_lesion_dice = fmean(
        metrics["lesion_dice"] for metrics in source_domain_metrics.values()
    )

    logger.info("Loaded checkpoint from epoch %d", checkpoint["epoch"])
    print_metrics(
        split_name="Source validation",
        metrics=source_val_metrics,
    )
    print_metrics(
        split_name="OOD development — BUS-UCLM",
        metrics=ood_metrics,
    )

    for domain, metrics in source_domain_metrics.items():
        print_metrics(
            split_name=f"Source validation — {domain}",
            metrics=metrics,
        )

    print(f"\nMacro source lesion Dice: {macro_source_lesion_dice:.4f}")

    evaluation_results = {
        "experiment_name": experiment_config.name,
        "seed": training_config.seed,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "source_validation": source_val_metrics,
        "source_domains": source_domain_metrics,
        "macro_source_lesion_dice": macro_source_lesion_dice,
        "ood_development": ood_metrics,
    }

    output_path = get_evaluation_result_path(
        experiment_name=experiment_config.name,
        seed=training_config.seed,
    )

    save_json(data=evaluation_results, path=output_path)
    logger.info("Evaluation results saved to %s", output_path)


if __name__ == "__main__":
    main()
