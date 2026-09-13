from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor


def create_domain_loaders(
    manifest: pd.DataFrame,
    domains: list[str],
    project_root: Path,
    adapters: dict[str, DatasetAdapter],
    preprocessor: SegmentationPreprocessor,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:

    loaders = {}

    for domain in domains:
        domain_manifest = manifest.loc[manifest["source_domain"] == domain].copy()

        if domain_manifest.empty:
            raise ValueError(f"No samples found for domain: {domain}")

        dataset = UltrasoundSegmentationDataset(
            manifest=domain_manifest,
            project_root=project_root,
            adapters=adapters,
            preprocessor=preprocessor,
        )

        loaders[domain] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return loaders
