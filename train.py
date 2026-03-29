"""Train PointNet on ModelNet10 for 3D point cloud classification."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ModelNet10Dataset, CLASSES
from models.pointnet import PointNetClassifier, feature_transform_regularization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet on ModelNet10")
    parser.add_argument("--data_root", type=str, default="data/ModelNet10",
                        help="Path to ModelNet10 dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=20,
                        help="StepLR scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="StepLR scheduler gamma")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--feat_reg_weight", type=float, default=0.001,
                        help="Feature transform regularization weight")
    parser.add_argument("--feature_transform", action="store_true", default=True)
    parser.add_argument("--no_feature_transform", dest="feature_transform",
                        action="store_false")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 for Windows/Colab compatibility)")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    feat_reg_weight: float,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in tqdm(loader, desc="Train", leave=False):
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _, feat_transform = model(points)

        loss = criterion(logits, labels)
        if feat_transform is not None:
            loss += feat_reg_weight * feature_transform_regularization(feat_transform)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in tqdm(loader, desc="Eval", leave=False):
        points, labels = points.to(device), labels.to(device)
        logits, _, _ = model(points)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def plot_training_curves(history: dict, save_path: str) -> None:
    """Plot and save training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"])
    ax2.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5,
                label=f"Best: {best_acc:.1%} (epoch {best_epoch})")
    ax2.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ModelNet10Dataset(
        root=args.data_root, split="train",
        num_points=args.num_points, augment=True,
    )
    test_dataset = ModelNet10Dataset(
        root=args.data_root, split="test",
        num_points=args.num_points, augment=False,
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # Model
    model = PointNetClassifier(
        num_classes=len(CLASSES),
        feature_transform=args.feature_transform,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} (lr={lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            args.feat_reg_weight, device,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }
            save_path = Path(args.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, str(save_path))
            print(f"  -> New best model saved! (acc={val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save training curves
    plot_training_curves(history, f"{args.results_dir}/training_curves.png")

    # Save training history
    with open(f"{args.results_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
