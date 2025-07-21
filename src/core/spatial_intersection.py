import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from collections import defaultdict

from ..data.observations import CameraPose, SpatialIntersectionData
from ..data.camera_models import CameraModel
from .geometry_utils import ray_from_camera, compute_ray_intersection_point, validate_ray_geometry


def triangulate_point(
    camera_poses: List[CameraPose],
    observations: List[npt.NDArray[np.float64]],
    camera_model: CameraModel
) -> npt.NDArray[np.float64]:
    """
    Triangulate a single 3D point from multiple camera observations.
    
    Uses least-squares intersection of rays from multiple cameras.
    
    Args:
        camera_poses: List of camera poses for cameras that observed this point
        observations: List of 2D pixel observations (x, y) for this point
        camera_model: Camera model for back-projection
        
    Returns:
        3D point estimate in world coordinates
        
    Raises:
        ValueError: If insufficient observations or invalid data
    """
    if len(camera_poses) < 2:
        raise ValueError(f"At least 2 camera observations required, got {len(camera_poses)}")
    
    if len(camera_poses) != len(observations):
        raise ValueError(f"Number of camera poses ({len(camera_poses)}) must match observations ({len(observations)})")
    
    # Compute rays from each camera
    rays = []
    for camera_pose, observation in zip(camera_poses, observations):
        try:
            origin, direction = ray_from_camera(camera_pose, observation, camera_model)
            if validate_ray_geometry(origin, direction):
                rays.append((origin, direction))
        except Exception as e:
            print(f"Warning: Failed to compute ray: {e}")
            continue
    
    if len(rays) < 2:
        raise ValueError(f"At least 2 valid rays required, got {len(rays)}")
    
    # For 2 rays, use direct intersection
    if len(rays) == 2:
        origin1, direction1 = rays[0]
        origin2, direction2 = rays[1]
        closest_point, _, _ = compute_ray_intersection_point(origin1, direction1, origin2, direction2)
        return closest_point
    
    # For 3+ rays, use least-squares intersection
    return _least_squares_intersection(rays)


def _least_squares_intersection(
    rays: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
) -> npt.NDArray[np.float64]:
    """
    Compute least-squares intersection of multiple rays.
    
    Uses the method of minimizing the sum of squared distances from the point to each ray.
    
    Args:
        rays: List of (origin, direction) tuples for each ray
        
    Returns:
        3D point that minimizes distance to all rays
    """
    num_rays = len(rays)
    
    # Build the linear system A * x = b
    # For each ray i: (I - d_i * d_i^T) * x = (I - d_i * d_i^T) * o_i
    A = np.zeros((3 * num_rays, 3))
    b = np.zeros(3 * num_rays)
    
    for i, (origin, direction) in enumerate(rays):
        # Projection matrix: P = I - d * d^T
        d = direction.reshape(3, 1)
        P = np.eye(3) - d @ d.T
        
        # Fill the linear system
        A[3*i:3*i+3] = P
        b[3*i:3*i+3] = P @ origin
    
    # Solve the least-squares problem: A^T * A * x = A^T * b
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Check condition number
    cond = np.linalg.cond(ATA)
    if cond > 1e12:
        print(f"Warning: Poorly conditioned system (condition number: {cond:.2e})")
    
    # Solve using least-squares
    try:
        point_3d = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        point_3d = np.linalg.pinv(ATA) @ ATb
    
    return point_3d


def run_spatial_intersection(data: SpatialIntersectionData) -> Tuple[npt.NDArray[np.float64], dict[int, int]]:
    """
    Run spatial intersection on all points in the dataset.
    
    Groups observations by point ID and triangulates each point from its observations.
    
    Args:
        data: SpatialIntersectionData containing camera poses and observations
        
    Returns:
        Tuple of:
            - 3D points array of shape (N, 3) where N is the number of unique points
            - dense_index_mapping: dict mapping original point indices to dense indices (0..N-1)
        
    Raises:
        ValueError: If data is invalid or no valid triangulations possible
    """
    # Validate input data
    data.validate()
    
    if len(data.points_2d) == 0:
        raise ValueError("No observations provided for triangulation")
    
    # Group observations by point index
    point_observations: Dict[int, List[Tuple[int, npt.NDArray[np.float64]]]] = defaultdict(list)
    
    for obs in data.points_2d:
        point_observations[obs.point_index].append((obs.camera_index, obs.location))
    
    # Get the number of unique points
    unique_point_indices = sorted(point_observations.keys())
    num_points = len(unique_point_indices)
    
    print(f"Triangulating {num_points} points from {len(data.points_2d)} observations")
    
    # Mapping from original point index to dense index
    dense_index_mapping = {orig_idx: dense_idx for dense_idx, orig_idx in enumerate(unique_point_indices)}
    
    # Initialize results array
    points_3d = np.zeros((num_points, 3))
    successful_triangulations = 0
    
    # Triangulate each point
    for i, point_idx in enumerate(unique_point_indices):
        observations_for_point = point_observations[point_idx]
        
        if len(observations_for_point) < 2:
            print(f"Warning: Point {point_idx} has only {len(observations_for_point)} observation(s), skipping")
            continue
        
        # Extract camera poses and observations for this point
        camera_poses_for_point = []
        observations_for_point_2d = []
        
        for camera_idx, pixel_coords in observations_for_point:
            camera_poses_for_point.append(data.camera_poses[camera_idx])
            observations_for_point_2d.append(pixel_coords)
        
        # Triangulate this point
        try:
            point_3d = triangulate_point(
                camera_poses_for_point,
                observations_for_point_2d,
                data.camera_model
            )
            points_3d[i] = point_3d
            successful_triangulations += 1
            
        except Exception as e:
            print(f"Warning: Failed to triangulate point {point_idx}: {e}")
            # Use a fallback estimate (e.g., origin)
            points_3d[i] = np.array([0.0, 0.0, 0.0])
    
    print(f"Successfully triangulated {successful_triangulations}/{num_points} points")
    
    return points_3d, dense_index_mapping


def compute_triangulation_quality(
    data: SpatialIntersectionData,
    points_3d: npt.NDArray[np.float64]
) -> Dict[str, float]:
    """
    Compute quality metrics for triangulation results.
    
    Args:
        data: Original spatial intersection data
        points_3d: Triangulated 3D points
        
    Returns:
        Dictionary containing quality metrics
    """
    # Group observations by point index
    point_observations: Dict[int, List[Tuple[int, npt.NDArray[np.float64]]]] = defaultdict(list)
    
    for obs in data.points_2d:
        point_observations[obs.point_index].append((obs.camera_index, obs.location))
    
    # Compute reprojection errors
    reprojection_errors = []
    
    for point_idx, observations in point_observations.items():
        if point_idx >= len(points_3d):
            continue
            
        point_3d_world = points_3d[point_idx]
        
        for camera_idx, pixel_obs in observations:
            camera_pose = data.camera_poses[camera_idx]
            
            # Transform point to camera coordinates
            point_cam = camera_pose.rotation @ (point_3d_world - camera_pose.translation)
            
            # Project to 2D
            if point_cam[2] > 0:  # Point is in front of camera
                projected_2d = data.camera_model.project(point_cam.reshape(1, 3))[0]
                error = np.linalg.norm(projected_2d - pixel_obs)
                reprojection_errors.append(error)
    
    if not reprojection_errors:
        return {
            "mean_error": float('inf'), 
            "std_error": float('inf'), 
            "max_error": float('inf'),
            "median_error": float('inf')
        }
    
    reprojection_errors = np.array(reprojection_errors)
    
    return {
        "mean_error": float(np.mean(reprojection_errors)),
        "std_error": float(np.std(reprojection_errors)),
        "max_error": float(np.max(reprojection_errors)),
        "median_error": float(np.median(reprojection_errors))
    }


def print_triangulation_summary(
    data: SpatialIntersectionData,
    points_3d: npt.NDArray[np.float64],
    quality_metrics: Dict[str, float]
) -> None:
    """
    Print a summary of triangulation results.
    
    Args:
        data: Original spatial intersection data
        points_3d: Triangulated 3D points
        quality_metrics: Quality metrics from compute_triangulation_quality
    """
    print("=" * 60)
    print("Spatial Intersection Results Summary")
    print("=" * 60)
    
    print(f"Input data:")
    print(f"  Cameras: {len(data.camera_poses)}")
    print(f"  Observations: {len(data.points_2d)}")
    
    print(f"\nTriangulation results:")
    print(f"  3D points: {points_3d.shape[0]}")
    
    # Point statistics
    if points_3d.shape[0] > 0:
        print(f"  Position range: X[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
        print(f"                  Y[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
        print(f"                  Z[{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")
    
    print(f"\nQuality metrics:")
    print(f"  Mean reprojection error: {quality_metrics['mean_error']:.3f} pixels")
    print(f"  Std reprojection error: {quality_metrics['std_error']:.3f} pixels")
    print(f"  Max reprojection error: {quality_metrics['max_error']:.3f} pixels")
    print(f"  Median reprojection error: {quality_metrics['median_error']:.3f} pixels")
    
    print("=" * 60) 