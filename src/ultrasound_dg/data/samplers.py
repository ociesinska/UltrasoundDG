import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def create_domain_balanced_sampler(
    train_manifest: pd.DataFrame,
    generator: torch.Generator,
) -> WeightedRandomSampler:
    """Sample training domains with equal probability."""

    domain_counts = train_manifest["source_domain"].value_counts()

    sample_weights = (
        train_manifest["source_domain"]
        .map(lambda domain: 1.0 / domain_counts[domain])
        .to_numpy()
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(train_manifest),
        replacement=True,
        generator=generator,
    )
