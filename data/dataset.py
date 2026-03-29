"""PyTorch Dataset for ModelNet10 point clouds."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

import trimesh


# ModelNet10 class names in alphabetical order
CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


def read_off(filepath: str) -> trimesh.Trimesh:
    """Read an .off mesh file, handling the ModelNet header quirk.

    Some ModelNet .off files have 'OFF' and vertex/face counts on the same line
    (e.g., 'OFF8326 2772 0') instead of separate lines. This function handles both.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

        if first_line == "OFF":
            # Standard format: counts on next line
            pass
        elif first_line.startswith("OFF"):
            # Quirky format: 'OFF' followed by counts on same line
            # Rewrite to a temp-fixed format for trimesh
            counts = first_line[3:].strip()
            rest = f.read()
            # Use trimesh to load from the fixed string
            lines = f"OFF\n{counts}\n{rest}"
            mesh = trimesh.load(
                trimesh.util.wrap_as_stream(lines),
                file_type="off",
            )
            return mesh
        else:
            raise ValueError(f"Not a valid OFF file: {filepath}")

    # Standard file — load directly
    mesh = trimesh.load(str(filepath), file_type="off")
    return mesh


def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int = 1024) -> np.ndarray:
    """Sample points uniformly from mesh surface.

    Args:
        mesh: Input triangle mesh.
        num_points: Number of points to sample.

    Returns:
        Point cloud array of shape [num_points, 3].
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.array(points, dtype=np.float32)


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """Center to origin and scale to unit sphere.

    Args:
        points: Point cloud of shape [N, 3].

    Returns:
        Normalized point cloud of shape [N, 3].
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points


def augment_pointcloud(
    points: np.ndarray,
    rotation: bool = True,
    jitter: bool = True,
    dropout: bool = True,
) -> np.ndarray:
    """Apply data augmentation to a point cloud.

    Args:
        points: Point cloud of shape [N, 3].
        rotation: Random rotation around Y-axis.
        jitter: Add small Gaussian noise.
        dropout: Randomly drop and duplicate points.

    Returns:
        Augmented point cloud of shape [N, 3].
    """
    if rotation:
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0    ],
            [-sin_t, 0, cos_t],
        ], dtype=np.float32)
        points = points @ rotation_matrix.T

    if jitter:
        noise = np.clip(
            np.random.normal(0, 0.02, size=points.shape),
            -0.05, 0.05,
        ).astype(np.float32)
        points = points + noise

    if dropout:
        n = len(points)
        num_drop = int(0.05 * n)
        drop_idx = np.random.choice(n, num_drop, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop_idx] = False
        kept = points[keep_mask]
        # Duplicate random kept points to restore count
        dup_idx = np.random.choice(len(kept), num_drop, replace=True)
        points = np.concatenate([kept, kept[dup_idx]], axis=0)

    return points


class ModelNet10Dataset(Dataset):
    """PyTorch Dataset for ModelNet10 point cloud classification.

    Reads .off mesh files, samples surface points, normalizes,
    and optionally applies data augmentation.

    Args:
        root: Path to ModelNet10 directory.
        split: 'train' or 'test'.
        num_points: Number of points to sample per mesh.
        augment: Whether to apply data augmentation (only for training).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_points: int = 1024,
        augment: bool = False,
    ):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"

        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment

        self.samples: list[Tuple[Path, int]] = []
        for cls_name in CLASSES:
            cls_dir = self.root / cls_name / split
            if not cls_dir.exists():
                continue
            label = CLASS_TO_IDX[cls_name]
            for off_file in sorted(cls_dir.glob("*.off")):
                self.samples.append((off_file, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .off files found in {self.root} for split='{split}'. "
                "Did you download the dataset? Run: python data/download_modelnet.py"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        # Load mesh and sample points
        mesh = read_off(str(filepath))
        points = sample_points_from_mesh(mesh, self.num_points)

        # Normalize
        points = normalize_pointcloud(points)

        # Augment (training only)
        if self.augment:
            points = augment_pointcloud(points)

        # Convert to tensor [N, 3]
        points_tensor = torch.from_numpy(points).float()

        return points_tensor, label

    def get_class_name(self, label: int) -> str:
        return CLASSES[label]
