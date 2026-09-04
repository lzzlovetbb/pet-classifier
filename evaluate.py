"""Evaluate a saved checkpoint and optionally save a normalized confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data.dataset import build_pet_dataloaders, set_seed
from models.model import build_resnet18
from utils.metrics import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--save-confusion-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / "artifacts"
    figure_dir = output_dir / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, class_names, _ = build_pet_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(checkpoint.get("label_smoothing", 0.0)))

    metrics, y_true, y_pred = evaluate_model(
        model, test_loader, criterion, device, return_predictions=True
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["best_epoch"] = checkpoint.get("best_epoch")
    print(json.dumps(metrics, indent=2))
    (artifact_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if args.save_confusion_matrix:
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
        figure.savefig(figure_dir / "confusion_matrix.png", dpi=200)
        plt.close(figure)
        np.savetxt(artifact_dir / "confusion_matrix_counts.csv", raw_counts, delimiter=",", fmt="%d")


if __name__ == "__main__":
    main()
