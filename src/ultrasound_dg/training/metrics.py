import math

import torch


def logits_to_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to binary masks using a probability threshold."""

    probabilities = torch.sigmoid(logits)

    return (probabilities >= threshold).float()


def dice_score(
    predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute one smoothed Dice score per image."""

    dims = tuple(
        range(1, predictions.ndim)
    )  # predictions.shape: [B, 1, H, W], range: (1,2,3)
    intersection = (predictions * targets).sum(
        dim=dims
    )  # summing all mask pixels, per image not per batch
    # sum: [B]

    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + smooth) / (denominator + smooth)

    return dice  # dice per image


def iou_score(
    predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute one smoothed intersection-over-union score per image."""

    dims = tuple(range(1, predictions.ndim))

    intersection = (predictions * targets).sum(dim=dims)

    union = predictions.sum(dim=dims) + targets.sum(dim=dims) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def recall_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute pixel-level recall independently for each image."""

    dims = tuple(range(1, predictions.ndim))
    true_positives = (predictions * targets).sum(dim=dims)
    actual_positives = targets.sum(dim=dims)

    recall = (true_positives + smooth) / (actual_positives + smooth)

    return recall


def precision_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return pixel-level precision for each image."""
    dims = tuple(range(1, predictions.ndim))
    true_positives = (predictions * targets).sum(dim=dims)
    predicted_positives = predictions.sum(dim=dims)

    return torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives.clamp_min(eps),
        torch.zeros_like(true_positives),
    )


def false_positive_pixel_fraction(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return the fraction of all pixels that are false positives, per image."""
    false_positive_pixels = predictions.bool() & ~targets.bool()

    return (
        false_positive_pixels.float()
        .flatten(start_dim=1)  # [B, 1, H, W] → [B, H × W]
        .mean(dim=1)
    )


def missed_lesion_image_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return 1 for a lesion-positive image when the model predicts no lesion pixels, otherwise 0."""
    target_has_lesion = targets.bool().flatten(start_dim=1).any(dim=1)

    prediction_has_lesion = predictions.bool().flatten(start_dim=1).any(dim=1)

    prediction_is_normal = ~prediction_has_lesion

    return (target_has_lesion & prediction_is_normal).float()


def print_metrics(
    split_name: str,
    metrics: dict[str, float],
) -> None:
    """Print the standard segmentation metric set for one data split."""
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

    if "patient_id_coverage" not in metrics:
        return

    print(f"  patient_id_coverage:  {metrics['patient_id_coverage']:.2%}")

    if not math.isfinite(metrics["patient_macro_dice"]):
        print("  patient-macro metrics: N/A")
        return

    print(
        "  patient_macro_dice:   "
        f"{_format_finite(metrics['patient_macro_dice'], '.4f')}"
    )
    print(
        "  patient_macro_lesion_dice:      "
        f"{_format_finite(metrics['patient_macro_lesion_dice'], '.4f')}"
    )
    print(
        "  patient_macro_lesion_precision: "
        f"{_format_finite(metrics['patient_macro_lesion_precision'], '.4f')}"
    )
    print(
        "  patient_macro_lesion_recall:    "
        f"{_format_finite(metrics['patient_macro_lesion_recall'], '.4f')}"
    )
    print(
        "  patient_macro_lesion_iou:       "
        f"{_format_finite(metrics['patient_macro_lesion_iou'], '.4f')}"
    )
    print(
        "  patient_macro_lesion_miss_rate: "
        f"{_format_finite(metrics['patient_macro_lesion_miss_rate'], '.2%')}"
    )
    print(
        "  patient_macro_normal_fp_fraction:   "
        f"{_format_finite(metrics['patient_macro_normal_fp_fraction'], '.4%')}"
    )
    print(
        "  patient_macro_normal_fp_image_rate: "
        f"{_format_finite(metrics['patient_macro_normal_fp_image_rate'], '.2%')}"
    )


def _format_finite(value: float, format_spec: str) -> str:
    """Format a finite metric or return a readable unavailable marker."""
    return format(value, format_spec) if math.isfinite(value) else "N/A"
