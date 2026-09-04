"""Training-independent classification metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Return the number of examples whose true label is in the top-k logits."""
    k = min(k, logits.size(1))
    topk_predictions = logits.topk(k, dim=1).indices
    return int(topk_predictions.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


def plot_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> np.ndarray:
    """Save a row-normalized confusion matrix and return its raw counts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_counts = confusion_matrix(y_true, y_pred)
    row_totals = raw_counts.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        raw_counts,
        row_totals,
        out=np.zeros_like(raw_counts, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(18, 16))
    display = ConfusionMatrixDisplay(normalized_matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", colorbar=True, xticks_rotation=90, values_format=".0%")
    axis.set_title("37-class normalized confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return raw_counts


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    """Evaluate with model.eval() and no gradients, preventing train-time behavior."""
    model.eval()
    total_examples = 0
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    targets_list: list[np.ndarray] = []
    predictions_list: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        top1_correct += topk_accuracy(logits, targets, k=1)
        top5_correct += topk_accuracy(logits, targets, k=5)

        if return_predictions:
            targets_list.append(targets.cpu().numpy())
            predictions_list.append(logits.argmax(dim=1).cpu().numpy())

    if total_examples == 0:
        raise RuntimeError("The evaluation loader was empty.")

    y_true = np.concatenate(targets_list) if return_predictions else None
    y_pred = np.concatenate(predictions_list) if return_predictions else None
    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if return_predictions
        else float("nan")
    )

    metrics = {
        "loss": total_loss / total_examples,
        "top1_accuracy": 100.0 * top1_correct / total_examples,
        "top5_accuracy": 100.0 * top5_correct / total_examples,
        "macro_f1": macro_f1,
    }
    return metrics, y_true, y_pred
