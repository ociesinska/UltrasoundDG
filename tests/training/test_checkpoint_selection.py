from pathlib import Path

from ultrasound_dg.training.checkpoint_selection import (
    CheckpointCandidate,
    eligible_checkpoint_candidates,
    select_checkpoint_candidate,
)


def _candidate(
    epoch: int,
    macro_dice: float,
    normal_fp_fraction: float,
) -> CheckpointCandidate:
    """Build a lightweight checkpoint candidate for selection tests."""
    return CheckpointCandidate(
        path=Path(f"epoch_{epoch}.pt"),
        epoch=epoch,
        macro_lesion_dice=macro_dice,
        normal_fp_fraction=normal_fp_fraction,
    )


def test_eligible_candidates_respect_dice_tolerance() -> None:
    """Keep only checkpoints sufficiently close to the best lesion Dice."""
    candidates = [
        _candidate(epoch=1, macro_dice=0.80, normal_fp_fraction=0.01),
        _candidate(epoch=2, macro_dice=0.82, normal_fp_fraction=0.02),
        _candidate(epoch=3, macro_dice=0.83, normal_fp_fraction=0.03),
    ]

    eligible = eligible_checkpoint_candidates(candidates, tolerance=0.01)

    assert [candidate.epoch for candidate in eligible] == [2, 3]


def test_selection_minimizes_normal_fp_within_dice_tolerance() -> None:
    """Prefer lower normal FP only among sufficiently accurate checkpoints."""
    candidates = [
        _candidate(epoch=1, macro_dice=0.80, normal_fp_fraction=0.001),
        _candidate(epoch=2, macro_dice=0.825, normal_fp_fraction=0.02),
        _candidate(epoch=3, macro_dice=0.83, normal_fp_fraction=0.03),
    ]

    selected = select_checkpoint_candidate(candidates, tolerance=0.01)

    assert selected.epoch == 2


def test_selection_falls_back_to_best_dice_without_normal_metrics() -> None:
    """Select the highest Dice when validation has no normal cases."""
    candidates = [
        _candidate(epoch=1, macro_dice=0.82, normal_fp_fraction=float("nan")),
        _candidate(epoch=2, macro_dice=0.84, normal_fp_fraction=float("nan")),
        _candidate(epoch=3, macro_dice=0.83, normal_fp_fraction=float("nan")),
    ]

    selected = select_checkpoint_candidate(candidates, tolerance=0.02)

    assert selected.epoch == 2
