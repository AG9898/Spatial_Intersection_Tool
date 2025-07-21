import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any


def print_intersection_summary(
    points_3d: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Print a comprehensive summary of intersection results.
    
    Args:
        points_3d: 3D points array (Nx3)
        errors: 1D array of reprojection errors (pixels)
        title: Optional title for the summary
    """
    if title:
        print("=" * 60)
        print(title)
        print("=" * 60)
    else:
        print("=" * 60)
        print("Spatial Intersection Summary")
        print("=" * 60)
    
    # Points statistics
    print(f"\n📊 3D Points Statistics:")
    print(f"  Number of points: {points_3d.shape[0]}")
    
    if points_3d.shape[0] > 0:
        # Bounding box
        x_min, x_max = points_3d[:, 0].min(), points_3d[:, 0].max()
        y_min, y_max = points_3d[:, 1].min(), points_3d[:, 1].max()
        z_min, z_max = points_3d[:, 2].min(), points_3d[:, 2].max()
        
        print(f"  Bounding box:")
        print(f"    X: [{x_min:.3f}, {x_max:.3f}] (range: {x_max - x_min:.3f})")
        print(f"    Y: [{y_min:.3f}, {y_max:.3f}] (range: {y_max - y_min:.3f})")
        print(f"    Z: [{z_min:.3f}, {z_max:.3f}] (range: {z_max - z_min:.3f})")
        
        # Center of mass
        center = np.mean(points_3d, axis=0)
        print(f"  Center of mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        
        # Spread statistics
        std_dev = np.std(points_3d, axis=0)
        print(f"  Standard deviation: ({std_dev[0]:.3f}, {std_dev[1]:.3f}, {std_dev[2]:.3f})")
    
    # Error statistics
    print(f"\n📈 Reprojection Error Statistics:")
    if len(errors) > 0:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        print(f"  Number of errors: {len(errors)}")
        print(f"  Mean error: {mean_error:.3f} pixels")
        print(f"  Standard deviation: {std_error:.3f} pixels")
        print(f"  Median error: {median_error:.3f} pixels")
        print(f"  Min error: {min_error:.3f} pixels")
        print(f"  Max error: {max_error:.3f} pixels")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"  Percentiles:")
        for p in percentiles:
            value = np.percentile(errors, p)
            print(f"    {p}%: {value:.3f} pixels")
        
        # Quality assessment
        print(f"\n✅ Quality Assessment:")
        if mean_error < 1.0:
            print(f"  ✓ Excellent accuracy (mean error < 1 pixel)")
        elif mean_error < 2.0:
            print(f"  ✓ Good accuracy (mean error < 2 pixels)")
        elif mean_error < 5.0:
            print(f"  ⚠ Acceptable accuracy (mean error < 5 pixels)")
        else:
            print(f"  ❌ Poor accuracy (mean error >= 5 pixels)")
        
        if std_error < mean_error * 0.5:
            print(f"  ✓ Consistent errors (low standard deviation)")
        else:
            print(f"  ⚠ Inconsistent errors (high standard deviation)")
        
        if max_error < mean_error * 3:
            print(f"  ✓ No extreme outliers")
        else:
            print(f"  ⚠ Some extreme outliers detected")
    else:
        print(f"  No error data available")
    
    print("=" * 60)


def print_detailed_statistics(
    points_3d: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    camera_poses: Optional[list] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print detailed statistics for intersection results.
    
    Args:
        points_3d: 3D points array (Nx3)
        errors: 1D array of reprojection errors (pixels)
        camera_poses: Optional list of camera poses for additional statistics
        additional_info: Optional dictionary with additional information
    """
    print("=" * 80)
    print("DETAILED SPATIAL INTERSECTION STATISTICS")
    print("=" * 80)
    
    # Basic summary
    print_intersection_summary(points_3d, errors, title=None)
    
    # Camera statistics
    if camera_poses is not None:
        print(f"\n📷 Camera Statistics:")
        print(f"  Number of cameras: {len(camera_poses)}")
        
        # Camera positions
        positions = np.array([pose.translation for pose in camera_poses])
        print(f"  Camera positions:")
        print(f"    X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print(f"    Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print(f"    Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        # Camera spacing
        if len(positions) > 1:
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            print(f"  Camera spacing:")
            print(f"    Mean distance: {np.mean(distances):.3f}")
            print(f"    Min distance: {np.min(distances):.3f}")
            print(f"    Max distance: {np.max(distances):.3f}")
    
    # Additional information
    if additional_info:
        print(f"\n📋 Additional Information:")
        for key, value in additional_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # Point density analysis
    if points_3d.shape[0] > 0:
        print(f"\n🎯 Point Density Analysis:")
        
        # Calculate point density in different regions
        center = np.mean(points_3d, axis=0)
        distances_from_center = np.linalg.norm(points_3d - center, axis=1)
        
        print(f"  Distance from center:")
        print(f"    Mean: {np.mean(distances_from_center):.3f}")
        print(f"    Std: {np.std(distances_from_center):.3f}")
        print(f"    Min: {np.min(distances_from_center):.3f}")
        print(f"    Max: {np.max(distances_from_center):.3f}")
        
        # Point distribution in octants
        octant_counts = np.zeros(8)
        for point in points_3d:
            octant = 0
            if point[0] >= center[0]: octant += 1
            if point[1] >= center[1]: octant += 2
            if point[2] >= center[2]: octant += 4
            octant_counts[octant] += 1
        
        print(f"  Point distribution in octants:")
        for i, count in enumerate(octant_counts):
            percentage = (count / len(points_3d)) * 100
            print(f"    Octant {i}: {count} points ({percentage:.1f}%)")
    
    print("=" * 80)


def print_comparison_summary(
    points_3d_true: npt.NDArray[np.float64],
    points_3d_estimated: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Print comparison summary between true and estimated 3D points.
    
    Args:
        points_3d_true: True 3D points (Nx3)
        points_3d_estimated: Estimated 3D points (Nx3)
        title: Optional title for the summary
    """
    if title:
        print("=" * 60)
        print(title)
        print("=" * 60)
    else:
        print("=" * 60)
        print("True vs Estimated 3D Points Comparison")
        print("=" * 60)
    
    if points_3d_true.shape[0] != points_3d_estimated.shape[0]:
        print("Warning: Different number of true and estimated points")
        return
    
    # Calculate errors
    errors_3d = np.linalg.norm(points_3d_estimated - points_3d_true, axis=1)
    
    print(f"\n📊 3D Reconstruction Accuracy:")
    print(f"  Number of points: {points_3d_true.shape[0]}")
    print(f"  Mean 3D error: {np.mean(errors_3d):.6f} units")
    print(f"  Std 3D error: {np.std(errors_3d):.6f} units")
    print(f"  Median 3D error: {np.median(errors_3d):.6f} units")
    print(f"  Min 3D error: {np.min(errors_3d):.6f} units")
    print(f"  Max 3D error: {np.max(errors_3d):.6f} units")
    
    # Component-wise errors
    component_errors = np.abs(points_3d_estimated - points_3d_true)
    print(f"\n📐 Component-wise errors:")
    print(f"  X-axis: mean={np.mean(component_errors[:, 0]):.6f}, std={np.std(component_errors[:, 0]):.6f}")
    print(f"  Y-axis: mean={np.mean(component_errors[:, 1]):.6f}, std={np.std(component_errors[:, 1]):.6f}")
    print(f"  Z-axis: mean={np.mean(component_errors[:, 2]):.6f}, std={np.std(component_errors[:, 2]):.6f}")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n📈 3D Error percentiles:")
    for p in percentiles:
        value = np.percentile(errors_3d, p)
        print(f"  {p}%: {value:.6f} units")
    
    # Quality assessment
    print(f"\n✅ 3D Reconstruction Quality:")
    mean_error = np.mean(errors_3d)
    if mean_error < 0.01:
        print(f"  ✓ Excellent 3D accuracy (mean error < 0.01 units)")
    elif mean_error < 0.1:
        print(f"  ✓ Good 3D accuracy (mean error < 0.1 units)")
    elif mean_error < 1.0:
        print(f"  ⚠ Acceptable 3D accuracy (mean error < 1.0 units)")
    else:
        print(f"  ❌ Poor 3D accuracy (mean error >= 1.0 units)")
    
    print("=" * 60)


def print_intersection_report(
    points_3d: npt.NDArray[np.float64],
    errors: npt.NDArray[np.float64],
    camera_poses: Optional[list] = None,
    processing_time: Optional[float] = None,
    algorithm_info: Optional[str] = None
) -> None:
    """
    Print a comprehensive intersection report.
    
    Args:
        points_3d: 3D points array (Nx3)
        errors: 1D array of reprojection errors (pixels)
        camera_poses: Optional list of camera poses
        processing_time: Optional processing time in seconds
        algorithm_info: Optional information about the algorithm used
    """
    print("=" * 80)
    print("SPATIAL INTERSECTION REPORT")
    print("=" * 80)
    
    # Algorithm information
    if algorithm_info:
        print(f"\n🔧 Algorithm: {algorithm_info}")
    
    # Processing time
    if processing_time is not None:
        print(f"\n⏱️ Processing time: {processing_time:.3f} seconds")
    
    # Detailed statistics
    print_detailed_statistics(points_3d, errors, camera_poses)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  • Successfully triangulated {points_3d.shape[0]} 3D points")
    if len(errors) > 0:
        print(f"  • Mean reprojection error: {np.mean(errors):.3f} pixels")
        print(f"  • 95th percentile error: {np.percentile(errors, 95):.3f} pixels")
    
    if camera_poses:
        print(f"  • Used {len(camera_poses)} cameras for triangulation")
    
    print("=" * 80) 