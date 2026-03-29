"""PointNet architecture for 3D point cloud classification.

Implements the PointNet model from:
    Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification
    and Segmentation", CVPR 2017.

Key architectural ideas:
    - Shared MLPs (Conv1d) extract per-point features with weight sharing
    - T-Net learns spatial transformations for alignment invariance
    - Max pooling aggregates per-point features into a global descriptor,
      providing permutation invariance (order of points doesn't matter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial Transformer Network (T-Net).

    Predicts a k×k transformation matrix to align input points or features
    into a canonical space. This makes the network invariant to certain
    geometric transformations (rotation, translation).

    The output is initialized to the identity matrix so the network starts
    by learning small perturbations from no-transform.

    Args:
        k: Dimension of the transformation matrix (3 for input, 64 for features).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # Shared MLP: per-point feature extraction
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers after max pooling
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a k×k transformation matrix.

        Args:
            x: Input tensor of shape [B, k, N].

        Returns:
            Transformation matrix of shape [B, k, k].
        """
        batch_size = x.shape[0]

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Symmetric function: max pool across points → global feature
        x = torch.max(x, dim=2)[0]  # [B, 1024]

        # FC layers to predict transformation
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [B, k*k]

        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype)
        identity = identity.flatten().unsqueeze(0).repeat(batch_size, 1)
        x = x + identity

        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet feature encoder.

    Extracts a global feature vector from an input point cloud by:
    1. Aligning input with T-Net (3×3)
    2. Extracting per-point features with shared MLPs
    3. Aligning features with T-Net (64×64)
    4. Further per-point feature extraction
    5. Max pooling to get a permutation-invariant global feature

    Args:
        global_feat: If True, return only the global feature [B, 1024].
                     If False, return concatenated local+global features.
        feature_transform: If True, use the 64×64 feature T-Net.
    """

    def __init__(self, global_feat: bool = True, feature_transform: bool = True):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input transform (3×3)
        self.input_tnet = TNet(k=3)

        # Shared MLP 1: 3 → 64 → 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transform (64×64)
        if self.feature_transform:
            self.feat_tnet = TNet(k=64)

        # Shared MLP 2: 64 → 128 → 1024
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract global feature from point cloud.

        Args:
            x: Input point cloud of shape [B, N, 3].

        Returns:
            Tuple of:
                - Global feature [B, 1024] (or local+global if global_feat=False)
                - Input transformation matrix [B, 3, 3]
                - Feature transformation matrix [B, 64, 64] (or None)
        """
        batch_size, num_points, _ = x.shape

        # Transpose for Conv1d: [B, N, 3] → [B, 3, N]
        x = x.transpose(1, 2)

        # Input alignment via T-Net
        input_transform = self.input_tnet(x)  # [B, 3, 3]
        x = x.transpose(1, 2)                 # [B, N, 3]
        x = torch.bmm(x, input_transform)     # [B, N, 3]
        x = x.transpose(1, 2)                 # [B, 3, N]

        # Shared MLP 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))    # [B, 64, N]

        # Feature alignment via T-Net
        feat_transform = None
        if self.feature_transform:
            feat_transform = self.feat_tnet(x)  # [B, 64, 64]
            x = x.transpose(1, 2)               # [B, N, 64]
            x = torch.bmm(x, feat_transform)    # [B, N, 64]
            x = x.transpose(1, 2)               # [B, 64, N]

        # Save per-point features for potential segmentation use
        point_features = x  # [B, 64, N]

        # Shared MLP 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))     # [B, 1024, N]

        # Max pool across points — the key symmetric function
        # This makes the representation invariant to point ordering
        x = torch.max(x, dim=2)[0]              # [B, 1024]

        if self.global_feat:
            return x, input_transform, feat_transform

        # For segmentation: concatenate global feature with per-point features
        x = x.unsqueeze(2).repeat(1, 1, num_points)  # [B, 1024, N]
        x = torch.cat([point_features, x], dim=1)     # [B, 1088, N]
        return x, input_transform, feat_transform


class PointNetClassifier(nn.Module):
    """PointNet classifier for 3D object classification.

    Combines the PointNet encoder with a classification head
    (fully connected layers with dropout and batch normalization).

    Args:
        num_classes: Number of output classes (10 for ModelNet10).
        feature_transform: Whether to use the 64×64 feature T-Net.
    """

    def __init__(self, num_classes: int = 10, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.num_classes = num_classes

        self.encoder = PointNetEncoder(
            global_feat=True,
            feature_transform=feature_transform,
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify a batch of point clouds.

        Args:
            x: Input point clouds of shape [B, N, 3].

        Returns:
            Tuple of:
                - Class logits [B, num_classes]
                - Input transformation matrix [B, 3, 3]
                - Feature transformation matrix [B, 64, 64] (or None)
        """
        global_feat, input_transform, feat_transform = self.encoder(x)

        # Classification head with dropout for regularization
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Raw logits — use CrossEntropyLoss (includes softmax)

        return x, input_transform, feat_transform


def feature_transform_regularization(transform: torch.Tensor) -> torch.Tensor:
    """Regularization loss for the feature transformation matrix.

    Encourages the 64×64 feature transform to be close to orthogonal,
    which helps training stability. The loss is: ||I - A * A^T||^2

    Args:
        transform: Feature transformation matrix of shape [B, k, k].

    Returns:
        Scalar regularization loss.
    """
    batch_size, k, _ = transform.shape
    identity = torch.eye(k, device=transform.device, dtype=transform.dtype)
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

    diff = identity - torch.bmm(transform, transform.transpose(1, 2))
    loss = torch.mean(torch.norm(diff, dim=(1, 2)) ** 2)
    return loss


if __name__ == "__main__":
    # Quick test: verify shapes with a random input
    model = PointNetClassifier(num_classes=10, feature_transform=True)

    # Simulate a batch of 4 point clouds, each with 1024 points
    dummy_input = torch.randn(4, 1024, 3)
    logits, input_t, feat_t = model(dummy_input)

    print(f"Input shape:              {dummy_input.shape}")
    print(f"Output logits shape:      {logits.shape}")
    print(f"Input transform shape:    {input_t.shape}")
    print(f"Feature transform shape:  {feat_t.shape}")

    reg_loss = feature_transform_regularization(feat_t)
    print(f"Feature reg loss:         {reg_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
