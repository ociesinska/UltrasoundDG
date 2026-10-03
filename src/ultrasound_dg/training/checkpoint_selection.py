import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointCandidate:
    """Store metrics required for checkpoint selection."""

    path: Path
    epoch: int
    macro_lesion_dice: float
    normal_fp_fraction: float


def eligible_checkpoint_candidates(
    candidates: list[CheckpointCandidate], tolerance: float
) -> list[CheckpointCandidate]:
    """Return checkpoints sufficiently close to the best lesion Dice."""
    if not candidates:
        raise ValueError("Cannot select a checkpoint without candidates.")

    best_macro_dice = max(candidate.macro_lesion_dice for candidate in candidates)

    minimum_acceptable_dice = best_macro_dice - tolerance

    return [
        candidate
        for candidate in candidates
        if candidate.macro_lesion_dice >= minimum_acceptable_dice
    ]


def select_checkpoint_candidate(
    candidates: list[CheckpointCandidate], tolerance: float
) -> CheckpointCandidate:
    """
    Select a high-Dice checkpoint with the lowest normal false-positive area.

    Checkpoints within `tolerance` of the best macro lesion Dice are eligible.
    Among eligible checkpoints with a finite normal-case metric, the one with
    the lowest normal FP fraction is selected. Higher Dice breaks ties.
    """

    eligible = eligible_checkpoint_candidates(
        candidates=candidates, tolerance=tolerance
    )

    with_normal_metrics = [
        candidate
        for candidate in eligible
        if math.isfinite(candidate.normal_fp_fraction)
    ]

    if with_normal_metrics:
        return min(
            with_normal_metrics,
            key=lambda candidate: (
                candidate.normal_fp_fraction,
                -candidate.macro_lesion_dice,
            ),
        )

    # some datasets contain no normal images
    return max(eligible, key=lambda candidate: candidate.macro_lesion_dice)
