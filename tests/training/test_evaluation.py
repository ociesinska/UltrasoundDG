import math

import pytest
import torch

from ultrasound_dg.training.evaluation import patient_macro_average


def test_patient_macro_average_gives_each_patient_equal_weight() -> None:
    """Ensure patients, rather than their image counts, receive equal weight."""
    image_scores = torch.tensor([1.0, 1.0, 0.0])
    patient_keys = ["domain:patient_a", "domain:patient_a", "domain:patient_b"]

    result = patient_macro_average(image_scores, patient_keys)

    assert result == pytest.approx(0.5)
    assert result != pytest.approx(image_scores.mean().item())


def test_patient_macro_average_requires_complete_patient_ids() -> None:
    """Return an unavailable metric when any patient identifier is missing."""
    result = patient_macro_average(
        torch.tensor([0.8, 0.6]),
        ["domain:patient_a", ""],
    )

    assert math.isnan(result)


def test_patient_macro_average_rejects_misaligned_inputs() -> None:
    """Reject score and identifier sequences with different lengths."""
    with pytest.raises(ValueError, match="equal length"):
        patient_macro_average(
            torch.tensor([0.8, 0.6]),
            ["domain:patient_a"],
        )
