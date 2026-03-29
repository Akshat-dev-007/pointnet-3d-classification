"""Run inference on a single .off file using a trained PointNet model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import (
    CLASSES, read_off, sample_points_from_mesh, normalize_pointcloud,
)
from models.pointnet import PointNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PointNet single-file inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .off mesh file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--save_viz", type=str, default=None,
                        help="Path to save visualization (optional)")
    return parser.parse_args()


def predict(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single point cloud.

    Returns:
        Tuple of (class_probabilities, sorted_class_indices).
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
        logits, _, _ = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    sorted_idx = np.argsort(probs)[::-1]
    return probs, sorted_idx


def visualize_prediction(
    points: np.ndarray,
    predicted_class: str,
    confidence: float,
    save_path: str | None = None,
) -> None:
    """Visualize the point cloud colored by predicted class."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=2, c=points[:, 1], cmap="viridis", alpha=0.7,
    )

    ax.set_title(
        f"Predicted: {predicted_class} ({confidence:.1%})",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})

    model = PointNetClassifier(
        num_classes=len(CLASSES),
        feature_transform=saved_args.get("feature_transform", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load and preprocess mesh
    print(f"Loading mesh: {args.input}")
    mesh = read_off(args.input)
    points = sample_points_from_mesh(mesh, args.num_points)
    points = normalize_pointcloud(points)

    # Predict
    probs, sorted_idx = predict(model, points, device)

    print(f"\nTop-3 Predictions:")
    for rank, idx in enumerate(sorted_idx[:3], 1):
        print(f"  {rank}. {CLASSES[idx]:<15} {probs[idx]:.4f} ({probs[idx]:.1%})")

    # Visualize
    predicted_class = CLASSES[sorted_idx[0]]
    confidence = probs[sorted_idx[0]]

    save_path = args.save_viz or f"results/inference_{Path(args.input).stem}.png"
    visualize_prediction(points, predicted_class, confidence, save_path)


if __name__ == "__main__":
    main()
