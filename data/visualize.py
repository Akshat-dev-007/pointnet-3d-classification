"""Visualize point clouds from ModelNet10 using Open3D and Matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.dataset import ModelNet10Dataset, CLASSES


def plot_pointcloud_grid(
    dataset: ModelNet10Dataset,
    save_path: str = "assets/sample_pointclouds.png",
    samples_per_class: int = 1,
) -> None:
    """Create a 2x5 matplotlib grid showing one sample point cloud per class.

    Args:
        dataset: ModelNet10Dataset instance.
        save_path: Path to save the output image.
        samples_per_class: Number of samples to show per class (uses first one).
    """
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("ModelNet10 — Sample Point Clouds", fontsize=16, fontweight="bold")

    # Collect one sample per class
    class_samples: dict[int, np.ndarray] = {}
    for i in range(len(dataset)):
        points, label = dataset[i]
        if label not in class_samples:
            class_samples[label] = points.numpy()
        if len(class_samples) == len(CLASSES):
            break

    for idx, (label, points) in enumerate(sorted(class_samples.items())):
        ax = fig.add_subplot(2, 5, idx + 1, projection="3d")
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=1, c=points[:, 1], cmap="viridis", alpha=0.6,
        )
        ax.set_title(CLASSES[label], fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved point cloud grid to {save_path}")


def visualize_single_open3d(points: np.ndarray, title: str = "Point Cloud") -> None:
    """Visualize a single point cloud interactively with Open3D.

    Args:
        points: Point cloud array of shape [N, 3].
        title: Window title.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not available. Use matplotlib visualization instead.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color by height (Y-axis)
    colors = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-8)
    cmap = plt.cm.viridis(colors)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(cmap)

    o3d.visualization.draw_geometries([pcd], window_name=title, width=800, height=600)


def plot_class_distribution(dataset: ModelNet10Dataset, save_path: str = "assets/class_distribution.png") -> None:
    """Plot a bar chart of class distribution.

    Args:
        dataset: ModelNet10Dataset instance.
        save_path: Path to save the output image.
    """
    counts = [0] * len(CLASSES)
    for _, label in dataset.samples:
        counts[label] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASSES, counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"ModelNet10 Class Distribution ({dataset.split})", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution to {save_path}")


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parent / "ModelNet10"

    if not data_root.exists():
        print(f"Dataset not found at {data_root}. Run download_modelnet.py first.")
        exit(1)

    # Load dataset (no augmentation for visualization)
    train_dataset = ModelNet10Dataset(root=str(data_root), split="train", num_points=1024)
    test_dataset = ModelNet10Dataset(root=str(data_root), split="test", num_points=1024)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # Generate visualizations
    plot_pointcloud_grid(train_dataset, save_path="assets/sample_pointclouds.png")
    plot_class_distribution(train_dataset, save_path="assets/class_distribution_train.png")
    plot_class_distribution(test_dataset, save_path="assets/class_distribution_test.png")

    # Try Open3D interactive visualization for one sample
    points, label = train_dataset[0]
    print(f"\nShowing: {CLASSES[label]}")
    visualize_single_open3d(points.numpy(), title=f"ModelNet10: {CLASSES[label]}")
