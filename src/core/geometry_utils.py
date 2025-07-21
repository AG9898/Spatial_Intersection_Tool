import numpy as np
import numpy.typing as npt
from typing import Tuple

from ..data.observations import CameraPose
from ..data.camera_models import CameraModel


def ray_from_camera(
    camera_pose: CameraPose, 
    pixel: npt.NDArray[np.float64], 
    camera_model: CameraModel
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the 3D ray from camera pose and 2D pixel observation.
    
    The ray is defined by its origin (camera center) and direction vector in world coordinates.
    
    Args:
        camera_pose: Camera pose with rotation and translation
        pixel: 2D pixel coordinates (x, y) in image space
        camera_model: Camera model for back-projection
        
    Returns:
        Tuple of (origin, direction) where:
        - origin: 3D camera center in world coordinates
        - direction: Normalized 3D ray direction in world coordinates
        
    Raises:
        ValueError: If pixel coordinates are invalid
    """
    # Validate input
    if pixel.shape != (2,):
        raise ValueError(f"Pixel must be 2D vector, got shape {pixel.shape}")
    
    # Step 1: Back-project pixel to normalized camera coordinates
    # Remove principal point offset
    x_norm = (pixel[0] - camera_model.principal_point[0]) / camera_model.focal_length
    y_norm = (pixel[1] - camera_model.principal_point[1]) / camera_model.focal_length
    
    # Create normalized ray direction in camera coordinates
    ray_cam = np.array([x_norm, y_norm, 1.0])
    
    # Step 2: Transform ray direction to world coordinates
    # ray_world = R * ray_cam (transform from camera to world)
    ray_world = camera_pose.rotation @ ray_cam
    
    # Step 3: Normalize the direction vector
    direction = ray_world / np.linalg.norm(ray_world)
    
    # Step 4: Camera center is the translation in world coordinates
    origin = camera_pose.translation
    
    return origin, direction


def compute_ray_distance(
    origin1: npt.NDArray[np.float64],
    direction1: npt.NDArray[np.float64],
    origin2: npt.NDArray[np.float64],
    direction2: npt.NDArray[np.float64]
) -> float:
    """
    Compute the minimum distance between two 3D rays.
    
    Uses the formula for the shortest distance between two skew lines.
    
    Args:
        origin1: Origin of first ray
        direction1: Direction of first ray (normalized)
        origin2: Origin of second ray
        direction2: Direction of second ray (normalized)
        
    Returns:
        Minimum distance between the two rays
    """
    # Vector between ray origins
    w = origin2 - origin1
    
    # Cross product of ray directions
    cross_dirs = np.cross(direction1, direction2)
    
    # Denominator: ||d1 × d2||^2
    denom = np.dot(cross_dirs, cross_dirs)
    
    if denom < 1e-12:  # Rays are nearly parallel
        # Distance is the perpendicular distance from one ray to the other
        w_perp = w - np.dot(w, direction1) * direction1
        return np.linalg.norm(w_perp)
    
    # Distance formula: |w · (d1 × d2)| / ||d1 × d2||
    distance = abs(np.dot(w, cross_dirs)) / np.sqrt(denom)
    
    return distance


def compute_ray_intersection_point(
    origin1: npt.NDArray[np.float64],
    direction1: npt.NDArray[np.float64],
    origin2: npt.NDArray[np.float64],
    direction2: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], float, float]:
    """
    Compute the closest point between two 3D rays.
    
    Returns the midpoint of the shortest line segment connecting the rays.
    
    Args:
        origin1: Origin of first ray
        direction1: Direction of first ray (normalized)
        origin2: Origin of second ray
        direction2: Direction of second ray (normalized)
        
    Returns:
        Tuple of (closest_point, t1, t2) where:
        - closest_point: 3D point closest to both rays
        - t1: Parameter along first ray
        - t2: Parameter along second ray
    """
    # Vector between ray origins
    w = origin2 - origin1
    
    # Dot products
    a = np.dot(direction1, direction1)  # Should be 1.0 for normalized vectors
    b = np.dot(direction1, direction2)
    c = np.dot(direction2, direction2)  # Should be 1.0 for normalized vectors
    d = np.dot(direction1, w)
    e = np.dot(direction2, w)
    
    # Denominator
    denom = a * c - b * b
    
    if abs(denom) < 1e-12:  # Rays are nearly parallel
        # Use midpoint of origins as fallback
        closest_point = (origin1 + origin2) / 2.0
        return closest_point, 0.0, 0.0
    
    # Parameters along each ray
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    # Closest points on each ray
    point1 = origin1 + t1 * direction1
    point2 = origin2 + t2 * direction2
    
    # Midpoint is the closest point
    closest_point = (point1 + point2) / 2.0
    
    return closest_point, t1, t2


def validate_ray_geometry(
    origin: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64]
) -> bool:
    """
    Validate that a ray has proper geometric properties.
    
    Args:
        origin: Ray origin
        direction: Ray direction
        
    Returns:
        True if ray is valid, False otherwise
    """
    # Check shapes
    if origin.shape != (3,) or direction.shape != (3,):
        return False
    
    # Check that direction is not zero
    if np.linalg.norm(direction) < 1e-12:
        return False
    
    # Check that direction is approximately normalized
    direction_norm = np.linalg.norm(direction)
    if abs(direction_norm - 1.0) > 1e-6:
        return False
    
    return True 