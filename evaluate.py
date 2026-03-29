"""Evaluate trained PointNet model on ModelNet10 test set."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm

from data.dataset import ModelNet10Dataset, CLASSES
from models.pointnet import PointNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PointNet on ModelNet10")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="data/ModelNet10")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], save_path: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix — PointNet on ModelNet10",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Reconstruct model
    saved_args = checkpoint.get("args", {})
    feature_transform = saved_args.get("feature_transform", True)

    model = PointNetClassifier(
        num_classes=len(CLASSES),
        feature_transform=feature_transform,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', '?')}")

    # Load test dataset
    test_dataset = ModelNet10Dataset(
        root=args.data_root, split="test",
        num_points=args.num_points, augment=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc="Evaluating"):
            points = points.to(device)
            logits, _, _ = model(points)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc:.1%})")

    # Per-class accuracy
    print(f"\n{'Class':<15} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
    print("-" * 40)
    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        acc = (all_preds[mask] == i).mean()
        per_class_acc[cls] = float(acc)
        correct = (all_preds[mask] == i).sum()
        total = mask.sum()
        print(f"{cls:<15} {acc:>8.4f} {correct:>8} {total:>6}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, CLASSES, str(results_dir / "confusion_matrix.png"))

    # Save metrics
    report_dict = classification_report(
        all_labels, all_preds, target_names=CLASSES, digits=4, output_dict=True,
    )
    metrics = {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
