from pathlib import Path
from statistics import fmean

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
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor
from ultrasound_dg.data.splits import create_development_protocol
from ultrasound_dg.models.unet import create_unet
from ultrasound_dg.training.checkpoints import load_checkpoint
from ultrasound_dg.training.device import resolve_device
from ultrasound_dg.training.evaluation import evaluate_loader
from ultrasound_dg.training.losses import BCEDiceLoss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "all_samples.csv"
CONFIG_ROOT = PROJECT_ROOT / "src" / "ultrasound_dg" / "configs"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEVELOPMENT_CONFIG_PATH = CONFIG_ROOT / "development" / "v1.yaml"
PREPROCESSING_CONFIG_PATH = CONFIG_ROOT / "preprocessing" / "v1.yaml"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "training" / "baseline_v1.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "baseline_v1"


def print_metrics(
    split_name: str,
    metrics: dict[str, float],
) -> None:
    print(f"\n{split_name}")
    print(f"  loss:                 {metrics['loss']:.4f}")
    print(f"  dice:                 {metrics['dice']:.4f}")
    print(f"  lesion_dice:          {metrics['lesion_dice']:.4f}")
    print(f"  lesion_precision:     {metrics['lesion_precision']:.4f}")
    print(f"  lesion_recall:        {metrics['lesion_recall']:.4f}")
    print(f"  lesion_iou:           {metrics['lesion_iou']:.4f}")
    print(f"  lesion_miss_rate:     {metrics['lesion_miss_rate']:.2%}")
    print(f"  normal_fp_fraction:   {metrics['normal_fp_fraction']:.4%}")
    print(f"  normal_fp_image_rate: {metrics['normal_fp_image_rate']:.2%}")


def main() -> None:
    training_config = load_config(TRAINING_CONFIG_PATH, TrainingConfig)
    preprocessing_config = load_config(
        PREPROCESSING_CONFIG_PATH,
        SegmentationPreprocessingConfig,
    )
    development_config = load_config(
        DEVELOPMENT_CONFIG_PATH,
        DevelopmentConfig,
    )

    device = resolve_device(training_config.device)
    model = create_unet().to(device)

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

    checkpoint = load_checkpoint(
        path=CHECKPOINT_DIR / "best_source_val.pt",
        model=model,
        optimizer=None,
        device=device,
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

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
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


if __name__ == "__main__":
    main()
