# Spatial Intersection Tool

A modular Python tool for computing 3D point positions via spatial (forward) intersection from 2D image observations and known camera poses. Supports both synthetic datasets for validation and real photogrammetric datasets exported from COLMAP.

---

## Key Features

- **Synthetic and Real Dataset Support**: Generate synthetic scenes for validation or load real datasets from COLMAP outputs.
- **Modular, Extensible Codebase**: Clean separation of data, core logic, solvers, and visualization for easy extension.
- **Comprehensive Visualization Tools**: 3D scene plots (cameras, points, rays), error histograms, and summary statistics.
- **Quality Metrics**: Automatic computation of reprojection errors and intersection diagnostics.
- **CLI Interface**: Consistent command-line interface mirroring the Bundle Adjustment Tool for seamless workflow integration.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run with Synthetic Data (default)

```bash
python main.py --dataset synthetic
```

### Run with COLMAP Data

```bash
python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
```

### Command-Line Options

- `--dataset {synthetic,colmap}`: Select dataset type (default: synthetic)
- `--images_txt`: Path to COLMAP images.txt (required for colmap)
- `--points3D_txt`: Path to COLMAP points3D.txt (required for colmap)

---

## Outputs

- **3D Plots**: Visualize camera positions, frustums, rays, and triangulated points.
- **Error Histograms**: Analyze reprojection error distributions.
- **Console Summaries**: Detailed statistics and quality metrics for each run.

---

## Future Extensions

- **Robust Intersection**: RANSAC and outlier rejection for noisy real-world datasets.
- **Ground Control Point Integration**: Support for georeferencing and control adjustment.
- **Advanced Camera Models**: Radial distortion, multi-camera rigs, and more.

---

## Project Structure

```
src/
├── data/           # Data structures, I/O utilities
├── core/           # Geometry and intersection logic
├── solvers/        # Triangulation algorithms
├── visualizations/ # 3D plots and diagnostics
└── main.py         # CLI driver
```

---

## Role in Photogrammetric Pipeline

The Spatial Intersection Tool performs forward intersection of 2D observations to compute 3D points, serving as a foundational step in photogrammetric workflows. It is designed to complement the Bundle Adjustment Tool, sharing data structures and coding conventions for seamless integration into larger pipelines such as `Photogrammetry_Pipeline`.

---

## License

MIT License

---

*Built for photogrammetry, computer vision, and educational use.* 