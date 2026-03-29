# PointNet: 3D Point Cloud Classification from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Open3D](https://img.shields.io/badge/Open3D-0.17%2B-green)
![CUDA](https://img.shields.io/badge/CUDA-Supported-brightgreen)

A clean, from-scratch implementation of **PointNet** (Qi et al., CVPR 2017) for 3D object classification on the **ModelNet10** dataset. Built to demonstrate foundational understanding of 3D point cloud processing — no high-level libraries hiding the architecture.

## Architecture

```
Input Point Cloud [N×3]
         │
         ▼
┌─────────────────┐
│  T-Net (3×3)    │  ← Learns spatial alignment transformation
└────────┬────────┘
         ▼
   Matrix Multiply     (align input to canonical space)
         │
         ▼
┌─────────────────┐
│ Shared MLP      │  ← Conv1d(3→64→64), same weights per point
│ (64, 64)        │
└────────┬────────┘
         ▼
┌─────────────────┐
│  T-Net (64×64)  │  ← Feature space alignment (with regularization)
└────────┬────────┘
         ▼
   Matrix Multiply     (align features)
         │
         ▼
┌─────────────────┐
│ Shared MLP      │  ← Conv1d(64→128→1024)
│ (64, 128, 1024) │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Max Pooling   │  ← Symmetric function → permutation invariance
└────────┬────────┘
         ▼
  Global Feature [1024]
         │
         ▼
┌─────────────────┐
│ FC (512→256→k)  │  ← Classification head with dropout
└────────┬────────┘
         ▼
   Class Prediction
```

## Key Concepts Demonstrated

- **Point cloud as 3D data representation** — unordered sets of (x, y, z) coordinates sampled from object surfaces
- **Permutation invariance via symmetric functions** — max pooling aggregates N points into a single global feature, regardless of point ordering
- **Spatial alignment via learned transformations (T-Net)** — mini-networks predict rotation matrices to canonicalize input pose
- **Per-point feature extraction via shared MLPs** — Conv1d layers apply identical transformations to each point (weight sharing)
- **Feature transform regularization** — constrains the 64×64 feature transform to be near-orthogonal for training stability

## Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | ~91-93% |
| **Training Time** | ~15-20 min (T4 GPU) |
| **Parameters** | ~3.5M |
| **Input Points** | 1024 |

### Sample Point Clouds
![Sample Point Clouds](assets/sample_pointclouds.png)

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Critical Points Visualization
The network focuses on geometrically distinctive points (edges, corners) rather than flat surfaces:

![Critical Points](results/critical_points.png)

## Project Structure

```
pointnet-3d-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── download_modelnet.py    # Download & extract ModelNet10
│   ├── dataset.py              # PyTorch Dataset with augmentation
│   └── visualize.py            # Point cloud visualization utilities
├── models/
│   └── pointnet.py             # PointNet architecture from scratch
├── train.py                    # Training script with logging
├── evaluate.py                 # Evaluation with metrics & confusion matrix
├── inference.py                # Single-file inference with visualization
├── notebooks/
│   └── pointnet_walkthrough.ipynb  # Full walkthrough notebook
├── results/                    # Generated plots and metrics
├── checkpoints/                # Saved model weights
└── assets/                     # README images
```

## How to Run

### Setup
```bash
pip install -r requirements.txt
python data/download_modelnet.py
```

### Train
```bash
python train.py --epochs 50 --batch_size 32 --num_points 1024
```

### Evaluate
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### Inference on a Single File
```bash
python inference.py --input path/to/object.off --checkpoint checkpoints/best_model.pth
```

### Notebook Walkthrough
Open `notebooks/pointnet_walkthrough.ipynb` in Jupyter or Google Colab for a step-by-step guide through the full pipeline.

## Discussion

### Common Confusions
- **night_stand vs dresser** — both are box-shaped furniture with drawers; differ mainly in size
- **desk vs table** — similar flat-surface-with-legs geometry
- **bathtub vs bed** — elongated shapes with raised edges

These confusions arise because PointNet operates on **global features only** — it captures the overall shape but misses fine-grained local details that distinguish similar objects.

### Limitations of PointNet
PointNet treats each point independently before the max pool. It has no mechanism to capture **local geometric structures** (e.g., surface patches, local curvature). This means:
- Fine details that distinguish similar classes are lost
- The network cannot reason about point neighborhoods

### How PointNet++ Addresses This
PointNet++ (Qi et al., NeurIPS 2017) introduces **hierarchical feature learning**:
1. **Sampling** — select representative points via farthest point sampling
2. **Grouping** — find local neighborhoods around each point
3. **PointNet** — apply PointNet to each local group
4. Repeat at multiple scales for multi-resolution features

## References

- Qi, C.R., Su, H., Mo, K. and Guibas, L.J., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
- Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X. and Xiao, J., "3D ShapeNets: A Deep Representation for Volumetric Shapes", CVPR 2015.
- [ModelNet Dataset](https://modelnet.cs.princeton.edu/)

## Future Work

- Extend to **PointNet++** for hierarchical local feature learning
- **Part segmentation** on ShapeNet dataset
- Apply to real-world **LiDAR/depth sensor** data (e.g., KITTI, ScanNet)
- Integration with **3D reconstruction** pipelines
- Experiment with different numbers of input points (512 vs 1024 vs 2048)
