import torch


def logits_to_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:

    probabilities = torch.sigmoid(logits)

    return (probabilities >= threshold).float()


def dice_score(
    predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:

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
