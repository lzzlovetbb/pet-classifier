"""Train one baseline or one single-variable label-smoothing experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from data.dataset import build_pet_dataloaders, set_seed
from models.model import build_resnet18
from utils.metrics import evaluate_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch: training mode, clear gradients, backward, then update."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())

    return total_loss / total_examples, 100.0 * total_correct / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="Where Oxford-IIIT Pet is stored.")
    parser.add_argument("--output-dir", default="outputs", help="Where generated files are saved.")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    artifact_dir = output_dir / "artifacts"
    log_dir = output_dir / "runs" / f"{args.experiment_name}_seed{args.seed}"
    for directory in (checkpoint_dir, artifact_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, test_loader, class_names, splits = build_pet_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        download=not args.no_download,
        split_file=artifact_dir / f"splits_seed{args.seed}.json",
    )

    model = build_resnet18(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_validation_top1 = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = checkpoint_dir / f"{args.experiment_name}_best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_metrics, _, _ = evaluate_model(
            model, validation_loader, criterion, device, return_predictions=False
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1_accuracy": train_top1,
            "validation_loss": validation_metrics["loss"],
            "validation_top1_accuracy": validation_metrics["top1_accuracy"],
        }
        history.append(row)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation_metrics["loss"], epoch)
        writer.add_scalar("accuracy/train_top1", train_top1, epoch)
        writer.add_scalar("accuracy/validation_top1", validation_metrics["top1_accuracy"], epoch)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train acc {train_top1:.2f}% | "
            f"validation acc {validation_metrics['top1_accuracy']:.2f}%"
        )

        if validation_metrics["top1_accuracy"] > best_validation_top1:
            best_validation_top1 = validation_metrics["top1_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "seed": args.seed,
                    "label_smoothing": args.label_smoothing,
                    "best_epoch": best_epoch,
                    "best_validation_top1": best_validation_top1,
                },
                checkpoint_path,
            )

    writer.close()
    history_path = artifact_dir / f"{args.experiment_name}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, y_true, y_pred = evaluate_model(
        model, test_loader, criterion, device, return_predictions=True
    )
    test_metrics.update(
        {
            "experiment": args.experiment_name,
            "best_epoch": best_epoch,
            "best_validation_top1": best_validation_top1,
            "label_smoothing": args.label_smoothing,
            "split_sizes": {name: len(values) for name, values in splits.items()},
        }
    )

    metrics_path = artifact_dir / f"{args.experiment_name}_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    np.savez(artifact_dir / f"{args.experiment_name}_test_predictions.npz", y_true=y_true, y_pred=y_pred)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Top-1: {best_validation_top1:.2f}%")
    print(f"Test Top-1: {test_metrics['top1_accuracy']:.2f}%")
    print(f"Test Top-5: {test_metrics['top5_accuracy']:.2f}%")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Best model: {checkpoint_path}")


if __name__ == "__main__":
    main()

