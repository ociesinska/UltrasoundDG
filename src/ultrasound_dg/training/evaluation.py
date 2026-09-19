import math
from collections.abc import Mapping
from statistics import fmean

import torch
from torch import nn
from torch.utils.data import DataLoader

from ultrasound_dg.training.metrics import (
    dice_score,
    false_positive_pixel_fraction,
    iou_score,
    logits_to_predictions,
    missed_lesion_image_score,
    precision_score,
    recall_score,
)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    normal_fp_area_threshold: float = 0.001,
) -> dict[str, float]:
    """
    Evaluate a binary segmentation model over a complete DataLoader.

    Computes mean loss and Dice across all samples; Dice, precision, recall,
    IoU and complete-lesion miss rate for positive cases; and false-positive
    area statistics for normal cases. Predictions are obtained by applying
    sigmoid to model logits and thresholding them at `threshold`.

    Returns aggregated dataset-level metrics.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_dice_scores = []
    lesion_dice_scores = []
    lesion_precision_scores = []
    lesion_recall_scores = []
    lesion_iou_scores = []
    normal_fp_fractions = []
    lesion_miss_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            has_lesion = batch["has_lesion"].to(device, dtype=torch.bool)
            lesion_selector = has_lesion
            normal_selector = ~has_lesion

            logits = model(images)
            loss = loss_fn(logits, targets)

            predictions = logits_to_predictions(logits, threshold=threshold)

            miss_scores_batch = missed_lesion_image_score(predictions, targets)

            # One Dice score per image: shape [B]
            dice_scores_batch = dice_score(
                predictions,
                targets,
            )
            precision_scores_batch = precision_score(predictions, targets)
            recall_scores_batch = recall_score(predictions, targets)
            iou_scores_batch = iou_score(predictions, targets)

            all_dice_scores.append(dice_scores_batch.cpu())

            if lesion_selector.any():
                # Dice only for lesion-positive images
                lesion_dice_scores.append(dice_scores_batch[lesion_selector].cpu())
                lesion_precision_scores.append(
                    precision_scores_batch[lesion_selector].cpu()
                )
                lesion_recall_scores.append(recall_scores_batch[lesion_selector].cpu())
                lesion_iou_scores.append(iou_scores_batch[lesion_selector].cpu())

                # Lesion miss score only for lesion-positive images
                lesion_miss_scores.append(miss_scores_batch[lesion_selector].cpu())

            # FP fraction per image: shape [B]
            fp_fraction_batch = false_positive_pixel_fraction(predictions, targets)

            # Only meaningful here for normal images
            if normal_selector.any():
                normal_fp_fractions.append(fp_fraction_batch[normal_selector].cpu())

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    dice_scores = torch.cat(all_dice_scores)
    mean_dice = dice_scores.mean().item()

    if lesion_dice_scores:
        lesion_dice_scores_tensor = torch.cat(lesion_dice_scores)
        lesion_precision_scores_tensor = torch.cat(lesion_precision_scores)
        lesion_recall_scores_tensor = torch.cat(lesion_recall_scores)
        lesion_iou_scores_tensor = torch.cat(lesion_iou_scores)
        lesion_miss_scores_tensor = torch.cat(lesion_miss_scores)

        mean_lesion_dice = lesion_dice_scores_tensor.mean().item()
        mean_lesion_precision = lesion_precision_scores_tensor.mean().item()
        mean_lesion_recall = lesion_recall_scores_tensor.mean().item()
        mean_lesion_iou = lesion_iou_scores_tensor.mean().item()

        lesion_miss_rate = lesion_miss_scores_tensor.mean().item()
    else:
        mean_lesion_dice = float("nan")
        mean_lesion_precision = float("nan")
        mean_lesion_recall = float("nan")
        mean_lesion_iou = float("nan")
        lesion_miss_rate = float("nan")

    if normal_fp_fractions:
        normal_fp_fractions_tensor = torch.cat(normal_fp_fractions)

        mean_normal_fp_fraction = normal_fp_fractions_tensor.mean().item()

        # fraction of normal images with a predicted lesion area larger than a small threshold
        normal_fp_image_rate = (
            (normal_fp_fractions_tensor > normal_fp_area_threshold)
            .float()
            .mean()
            .item()
        )
    else:
        mean_normal_fp_fraction = float("nan")
        normal_fp_image_rate = float("nan")

    return {
        "loss": total_loss / total_samples,
        "dice": mean_dice,
        "lesion_dice": mean_lesion_dice,
        "lesion_precision": mean_lesion_precision,
        "lesion_recall": mean_lesion_recall,
        "lesion_iou": mean_lesion_iou,
        "lesion_miss_rate": lesion_miss_rate,
        "normal_fp_fraction": mean_normal_fp_fraction,
        "normal_fp_image_rate": normal_fp_image_rate,
    }


def macro_average_domain_metric(
    metrics_by_domain: Mapping[str, Mapping[str, float]],
    metric_name: str,
) -> float:
    if not metrics_by_domain:
        raise ValueError("Cannot calculate a macro average without domains.")

    values = []

    for domain, metrics in metrics_by_domain.items():
        if metric_name not in metrics:
            raise KeyError(f"Metric {metric_name!r} is missing for domain {domain!r}.")

        value = metrics[metric_name]

        if not math.isfinite(value):
            raise ValueError(
                f"Metric {metric_name!r} is not finite for domain {domain!r}: {value}"
            )

        values.append(value)

    return fmean(values)
