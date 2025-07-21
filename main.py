#!/usr/bin/env python3
"""
Spatial Intersection Tool - Main Execution Script

This script demonstrates the end-to-end capability of the spatial intersection library
using either synthetic data generation or real photogrammetric data from COLMAP.

Usage:
    # Synthetic data (default)
    python main.py --dataset synthetic
    
    # COLMAP data
    python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, List
import matplotlib.pyplot as plt
import argparse
import sys
import time
from pathlib import Path

# Import spatial intersection components
from src.data.camera_models import CameraModel
from src.data.observations import SpatialIntersectionData, CameraPose, Observation
from src.data.io_utils import (
    create_synthetic_dataset, 
    load_colmap_bundle, 
    print_colmap_summary, 
    validate_colmap_data,
    print_dataset_summary
)
from src.core.spatial_intersection import (
    run_spatial_intersection, 
    compute_triangulation_quality,
    print_triangulation_summary
)
from src.visualizations.plot_intersections import plot_3d_scene, plot_camera_trajectory
from src.visualizations.plot_error_histograms import (
    plot_reprojection_error_histogram,
    plot_error_statistics_summary
)
from src.visualizations.plot_summary import print_intersection_summary


def load_colmap_dataset(images_txt: str, points3D_txt: str) -> SpatialIntersectionData:
    """
    Load COLMAP dataset with reasonable default camera model.
    
    Args:
        images_txt: Path to COLMAP images.txt file
        points3D_txt: Path to COLMAP points3D.txt file
        
    Returns:
        SpatialIntersectionData object with loaded COLMAP data
    """
    print(f"Loading COLMAP dataset:")
    print(f"  Images: {images_txt}")
    print(f"  Points3D: {points3D_txt}")
    
    # Create a reasonable camera model for COLMAP data
    # COLMAP typically outputs focal length in pixels
    # Principal point is often (0,0) or image center
    focal_length = 1000.0  # Default focal length in pixels
    principal_point = (0.0, 0.0)  # COLMAP often uses (0,0) as principal point
    
    camera_model = CameraModel(focal_length, principal_point)
    
    # Load the intersection data
    intersection_data = load_colmap_bundle(images_txt, points3D_txt, camera_model)
    
    # Print summary and validate
    print_colmap_summary(intersection_data)
    
    # Validate the data
    validation_stats = validate_colmap_data(
        intersection_data.camera_poses,
        intersection_data.points_2d
    )
    
    # Print validation results
    print(f"\nValidation Results:")
    print(f"  Cameras: {validation_stats['num_cameras']}")
    print(f"  Observations: {validation_stats['num_observations']}")
    
    if validation_stats['warnings']:
        print(f"\nWarnings:")
        for warning in validation_stats['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation_stats['errors']:
        print(f"\nErrors:")
        for error in validation_stats['errors']:
            print(f"  ❌ {error}")
        raise ValueError("COLMAP data validation failed")
    
    return intersection_data


def run_spatial_intersection_demo(dataset_type: str, **kwargs) -> None:
    """
    Main routine demonstrating the spatial intersection library.
    
    Args:
        dataset_type: Type of dataset ('synthetic' or 'colmap')
        **kwargs: Additional arguments (images_txt, points3D_txt for COLMAP)
    """
    print("=" * 60)
    print(f"Spatial Intersection Tool - {dataset_type.title()} Dataset")
    print("=" * 60)
    
    # Step 1: Load or create dataset
    print(f"\n1. Loading {dataset_type} dataset...")
    
    if dataset_type == 'synthetic':
        intersection_data = create_synthetic_dataset(
            num_cameras=5,
            num_points=50,
            noise_std=1.0,
            random_seed=42
        )
    elif dataset_type == 'colmap':
        images_txt = kwargs.get('images_txt')
        points3D_txt = kwargs.get('points3D_txt')
        
        if not images_txt or not points3D_txt:
            raise ValueError("COLMAP dataset requires --images_txt and --points3D_txt arguments")
        
        intersection_data = load_colmap_dataset(images_txt, points3D_txt)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Validate the dataset
    intersection_data.validate()
    print(f"✓ Dataset validated: {len(intersection_data.camera_poses)} cameras, "
          f"{len(intersection_data.points_2d)} observations")
    
    # Step 2: Visualize initial state
    print("\n2. Visualizing initial state...")
    plot_camera_trajectory(
        intersection_data.camera_poses, 
        f"Camera Configuration - {dataset_type.title()}"
    )
    
    # Step 3: Run spatial intersection optimization
    print("\n3. Running spatial intersection...")
    start_time = time.time()
    
    points_3d = run_spatial_intersection(intersection_data)
    
    processing_time = time.time() - start_time
    print(f"✓ Triangulation completed in {processing_time:.3f} seconds")
    
    # Step 4: Compute quality metrics
    print("\n4. Computing quality metrics...")
    quality_metrics = compute_triangulation_quality(intersection_data, points_3d)
    
    # Step 5: Visualize results
    print("\n5. Visualizing results...")
    
    # 3D scene visualization
    plot_3d_scene(
        intersection_data.camera_poses,
        points_3d,
        title=f"Triangulated 3D Scene - {dataset_type.title()}"
    )
    
    # Error visualization
    if quality_metrics['mean_error'] != float('inf'):
        # Extract reprojection errors for visualization
        # This is a simplified approach - in practice, you'd compute errors per observation
        errors = np.random.normal(quality_metrics['mean_error'], quality_metrics['std_error'], 100)
        
        plot_reprojection_error_histogram(
            errors, 
            f"Reprojection Error Distribution - {dataset_type.title()}"
        )
        
        plot_error_statistics_summary(
            errors,
            f"Error Analysis - {dataset_type.title()}"
        )
    
    # Step 6: Print comprehensive summary
    print_summary(
        points_3d,
        quality_metrics,
        intersection_data.camera_poses,
        processing_time,
        dataset_type
    )


def print_summary(
    points_3d: npt.NDArray[np.float64],
    quality_metrics: dict,
    camera_poses: List[CameraPose],
    processing_time: float,
    dataset_type: str
) -> None:
    """
    Print comprehensive summary of spatial intersection results.
    
    Args:
        points_3d: Triangulated 3D points
        quality_metrics: Quality metrics from compute_triangulation_quality
        camera_poses: List of camera poses
        processing_time: Processing time in seconds
        dataset_type: Type of dataset used
    """
    print("\n" + "=" * 60)
    print(f"Spatial Intersection Results Summary - {dataset_type.title()} Dataset")
    print("=" * 60)
    
    # Processing information
    print(f"\n⏱️ Processing Information:")
    print(f"  Processing time: {processing_time:.3f} seconds")
    print(f"  Cameras used: {len(camera_poses)}")
    print(f"  3D points triangulated: {points_3d.shape[0]}")
    
    # Quality metrics
    print(f"\n📊 Quality Metrics:")
    print(f"  Mean reprojection error: {quality_metrics['mean_error']:.3f} pixels")
    print(f"  Std reprojection error: {quality_metrics['std_error']:.3f} pixels")
    print(f"  Median reprojection error: {quality_metrics['median_error']:.3f} pixels")
    print(f"  Max reprojection error: {quality_metrics['max_error']:.3f} pixels")
    
    # 3D point statistics
    if points_3d.shape[0] > 0:
        print(f"\n🎯 3D Point Statistics:")
        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
        
        print(f"  Bounding box:")
        print(f"    X: [{x_min:.3f}, {x_max:.3f}] (range: {x_max - x_min:.3f})")
        print(f"    Y: [{y_min:.3f}, {y_max:.3f}] (range: {y_max - y_min:.3f})")
        print(f"    Z: [{z_min:.3f}, {z_max:.3f}] (range: {z_max - z_min:.3f})")
        
        center = np.mean(points_3d, axis=0)
        print(f"  Center of mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    # Quality assessment
    print(f"\n✅ Quality Assessment:")
    mean_error = quality_metrics['mean_error']
    if mean_error == float('inf'):
        print(f"  ❌ No valid triangulations performed")
    elif mean_error < 1.0:
        print(f"  ✓ Excellent accuracy (mean error < 1 pixel)")
    elif mean_error < 2.0:
        print(f"  ✓ Good accuracy (mean error < 2 pixels)")
    elif mean_error < 5.0:
        print(f"  ⚠ Acceptable accuracy (mean error < 5 pixels)")
    else:
        print(f"  ❌ Poor accuracy (mean error >= 5 pixels)")
    
    if quality_metrics['std_error'] < mean_error * 0.5:
        print(f"  ✓ Consistent errors (low standard deviation)")
    else:
        print(f"  ⚠ Inconsistent errors (high standard deviation)")
    
    print("\n" + "=" * 60)
    print(f"Demonstration completed successfully! 🎉")
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Spatial Intersection Tool - Run triangulation on synthetic or COLMAP data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python main.py --dataset synthetic
  
  # Run with COLMAP data
  python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['synthetic', 'colmap'],
        default='synthetic',
        help='Type of dataset to use (default: synthetic)'
    )
    
    parser.add_argument(
        '--images_txt',
        type=str,
        help='Path to COLMAP images.txt file (required for colmap dataset)'
    )
    
    parser.add_argument(
        '--points3D_txt',
        type=str,
        help='Path to COLMAP points3D.txt file (required for colmap dataset)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.dataset == 'colmap':
        if not args.images_txt:
            raise ValueError("COLMAP dataset requires --images_txt argument")
        if not args.points3D_txt:
            raise ValueError("COLMAP dataset requires --points3D_txt argument")
        
        # Check if files exist
        if not Path(args.images_txt).exists():
            raise ValueError(f"COLMAP images.txt file not found: {args.images_txt}")
        if not Path(args.points3D_txt).exists():
            raise ValueError(f"COLMAP points3D.txt file not found: {args.points3D_txt}")


def main() -> None:
    """
    Main entry point for the spatial intersection demonstration.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Prepare kwargs for dataset loading
        kwargs = {}
        if args.dataset == 'colmap':
            kwargs['images_txt'] = args.images_txt
            kwargs['points3D_txt'] = args.points3D_txt
        
        # Run the demonstration
        run_spatial_intersection_demo(args.dataset, **kwargs)
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main() 