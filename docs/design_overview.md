# Spatial Intersection Tool — Design Overview

---

## 1️⃣ Purpose

The Spatial Intersection Tool performs forward intersection of 2D image observations to compute 3D point positions, given known camera intrinsics and poses. It is a core photogrammetric operation, enabling the reconstruction of 3D structure from multiple calibrated images.

This tool is designed to complement the Bundle Adjustment Tool, providing a complete photogrammetric workflow from initial intersection to global optimization.

---

## 2️⃣ Inputs & Outputs

**Inputs:**
- Camera intrinsics (focal length, principal point)
- Camera poses (rotation, translation)
- 2D image observations (pixel coordinates)
- (Optional) COLMAP-format files: `images.txt`, `points3D.txt`

**Outputs:**
- 3D point coordinates (world frame)
- Reprojection error diagnostics
- Visualization plots (3D scene, error histograms)
- Console summary statistics

---

## 3️⃣ System Architecture

```
src/
├── data/           # Data structures, I/O utilities
├── core/           # Geometry and intersection logic
├── solvers/        # Triangulation algorithms
├── visualizations/ # Plots and diagnostics
└── main.py         # CLI driver
```

- **src/data/**: Defines camera models, observation structures, and dataset I/O (synthetic and COLMAP).
- **src/core/**: Implements geometric computations (ray back-projection, least-squares intersection).
- **src/solvers/**: (Reserved for advanced/robust triangulation algorithms).
- **src/visualizations/**: 3D scene plots, error histograms, and summary reporting.
- **src/main.py**: Command-line interface for running intersection workflows.

---

## 4️⃣ CLI Design

- Mirrors the Bundle Adjustment Tool for consistency.
- Supports both synthetic and COLMAP datasets.
- Example usage:

```bash
python main.py --dataset synthetic
python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
```

- Outputs 3D plots, error histograms, and summary statistics to the console.

---

## 5️⃣ Scope

- **Focus:** Forward intersection (triangulation) of 3D points from 2D observations and known camera poses.
- **Not included:**
  - Initial camera calibration or pose estimation
  - Bundle adjustment (handled by the Bundle Adjustment Tool)
  - Ground control point adjustment (future extension)

---

## 6️⃣ Alignment with Bundle Adjustment Tool

- **Shared Data Structures:** Camera models, pose and observation classes, dataset containers.
- **Consistent Coding Conventions:** Type annotations, modular design, and documentation style.
- **Pipeline Compatibility:** Designed for integration into larger photogrammetric pipelines (e.g., `Photogrammetry_Pipeline`).
- **Visualization and Diagnostics:** Plots and summary statistics follow the same conventions for easy comparison and debugging.

---

## 7️⃣ Role in Photogrammetric Pipeline

The Spatial Intersection Tool is the first step in a complete photogrammetric workflow:

1. **Spatial Intersection:** Compute initial 3D points from 2D observations and known camera poses.
2. **Bundle Adjustment:** Refine camera poses and 3D points for global optimality (see Bundle Adjustment Tool).
3. **(Future) Control Adjustment:** Integrate ground control points for georeferencing.

---

*This design overview provides the technical foundation for understanding, extending, and integrating the Spatial Intersection Tool within modern photogrammetric pipelines.* 