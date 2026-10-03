import pandas as pd
import pytest
import torch

from ultrasound_dg.data.samplers import create_domain_balanced_sampler


def test_domain_balanced_sampler_assigns_equal_total_weight_per_domain() -> None:
    """Give each domain equal sampling mass despite unequal sample counts."""
    manifest = pd.DataFrame(
        {
            "source_domain": [
                "large_domain",
                "large_domain",
                "large_domain",
                "small_domain",
            ]
        }
    )
    generator = torch.Generator().manual_seed(7)

    sampler = create_domain_balanced_sampler(
        train_manifest=manifest,
        generator=generator,
    )

    weights = pd.Series(sampler.weights.tolist(), index=manifest.index)
    total_weights = weights.groupby(manifest["source_domain"]).sum()

    assert sampler.num_samples == len(manifest)
    assert sampler.replacement is True
    assert total_weights["large_domain"] == pytest.approx(1.0)
    assert total_weights["small_domain"] == pytest.approx(1.0)
