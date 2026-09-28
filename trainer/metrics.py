"""Legacy pixel-level metrics used by retained segmentation components.

Reader's guide: these metrics operate on image masks, not SonoRank's
finger-flexion events. The SonoRank training commands compute classification
or ranking metrics locally, while event-level evaluation lives under
``evaluation/``.
"""

import torch
import torch.nn as nn
import numpy as np


class IoU(nn.Module):
    """
    Intersection over Union (IoU) metric.
    Also known as Jaccard Index.
    """

    def __init__(self, num_classes=8, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Compute IoU.

        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)

        Returns:
            Mean IoU across all classes
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        iou_per_class = []

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union == 0:
                iou_per_class.append(torch.tensor(1.0, device=pred.device))
            else:
                iou_per_class.append(intersection / union)

        return torch.stack(iou_per_class).mean()


class DiceScore(nn.Module):
    """
    Dice Score metric.
    """

    def __init__(self, num_classes=8, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Compute Dice score.

        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)

        Returns:
            Mean Dice score across all classes
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        dice_per_class = []

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().float()
            cardinality = pred_mask.sum().float() + target_mask.sum().float()

            if cardinality == 0:
                dice_per_class.append(torch.tensor(1.0, device=pred.device))
            else:
                dice_per_class.append(2 * intersection / cardinality)

        return torch.stack(dice_per_class).mean()


class PixelAccuracy(nn.Module):
    """
    Pixel-wise accuracy metric.
    """

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Compute pixel accuracy.

        Args:
            pred: Predictions (B, C, H, W) or (B, H, W)
            target: Ground truth (B, H, W)

        Returns:
            Pixel accuracy
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            correct = (pred[mask] == target[mask]).sum().float()
            total = mask.sum().float()
        else:
            correct = (pred == target).sum().float()
            total = target.numel()

        return correct / total if total > 0 else torch.tensor(0.0, device=pred.device)


def compute_metrics(pred, target, num_classes=8):
    """
    Compute all metrics at once.

    Args:
        pred: Predictions (B, C, H, W)
        target: Ground truth (B, H, W)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    iou_metric = IoU(num_classes=num_classes)
    dice_metric = DiceScore(num_classes=num_classes)
    acc_metric = PixelAccuracy()

    with torch.no_grad():
        iou = iou_metric(pred, target)
        dice = dice_metric(pred, target)
        acc = acc_metric(pred, target)

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'pixel_acc': acc.item(),
    }


def compute_per_class_metrics(pred, target, num_classes=8):
    """
    Compute per-class IoU and Dice scores.

    Args:
        pred: Predictions (B, C, H, W) or (B, H, W)
        target: Ground truth (B, H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with per-class metrics
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    iou_per_class = []
    dice_per_class = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        cardinality = pred_mask.sum().float() + target_mask.sum().float()

        # IoU
        if union == 0:
            iou = 1.0
        else:
            iou = (intersection / union).item()
        iou_per_class.append(iou)

        # Dice
        if cardinality == 0:
            dice = 1.0
        else:
            dice = (2 * intersection / cardinality).item()
        dice_per_class.append(dice)

    return {
        'iou_per_class': iou_per_class,
        'dice_per_class': dice_per_class,
    }
