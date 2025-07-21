import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation

from .camera_models import CameraModel
from .observations import CameraPose, Observation, SpatialIntersectionData


def create_synthetic_dataset(
    num_cameras: int = 5,
    num_points: int = 50,
    noise_std: float = 1.0,
    random_seed: int = 42
) -> SpatialIntersectionData:
    """
    Create a synthetic spatial intersection dataset with known ground truth.
    
    Args:
        num_cameras: Number of cameras to create (default: 5)
        num_points: Number of 3D points to create (default: 50)
        noise_std: Standard deviation of Gaussian noise for observations (default: 1.0)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        SpatialIntersectionData instance with synthetic camera poses and observations
    """
    np.random.seed(random_seed)
    
    print(f"Creating synthetic dataset: {num_cameras} cameras, {num_points} points")
    
    # Create camera model
    focal_length = 1000.0
    principal_point = (640.0, 480.0)
    camera_model = CameraModel(focal_length, principal_point)
    
    # Create camera poses in a circular pattern for better coverage
    camera_poses = []
    
    for i in range(num_cameras):
        # Circular arrangement around the scene
        angle = i * 2 * np.pi / num_cameras
        radius = 12.0  # Closer to the scene
        
        # Translation: circular pattern with height variation
        translation = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            6.0 + np.sin(i * 0.5) * 0.5  # Lower height, less variation
        ])
        
        # Rotation: look towards center
        center_direction = np.array([0, 0, 0]) - translation
        center_direction = center_direction / np.linalg.norm(center_direction)
        
        # Create rotation matrix
        up = np.array([0, 0, 1])
        right = np.cross(center_direction, up)
        if np.linalg.norm(right) > 1e-8:
            right = right / np.linalg.norm(right)
            up = np.cross(right, center_direction)
        else:
            # Fallback if center_direction is parallel to up
            right = np.array([1, 0, 0])
            up = np.array([0, 1, 0])
        
        # The camera looks in the positive Z direction towards the center
        rotation = np.column_stack([right, up, center_direction])
        
        camera_poses.append(CameraPose(rotation, translation))
    
    # Create 3D points in a structured pattern
    points_3d = np.zeros((num_points, 3))
    
    # Create points in a structured pattern (sphere + some random points)
    n_structured = min(num_points, int(num_points * 0.8))
    n_random = num_points - n_structured
    
    # Structured points in a sphere centered at origin
    if n_structured > 0:
        # Generate points on a sphere surface
        phi = np.random.uniform(0, 2*np.pi, n_structured)
        theta = np.arccos(np.random.uniform(-1, 1, n_structured))
        radius = 6.0  # Smaller radius to ensure visibility
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        points_3d[:n_structured] = np.column_stack([x, y, z])
    
    # Random points for additional coverage (closer to origin)
    if n_random > 0:
        random_points = np.random.uniform(
            low=[-8, -8, -4],
            high=[8, 8, 4],
            size=(n_random, 3)
        )
        points_3d[n_structured:] = random_points
    
    # Generate observations by projecting points
    observations = []
    
    # For each 3D point, find which cameras can see it
    for point_idx in range(num_points):
        point_3d_world = points_3d[point_idx]
        visible_cameras = []
        
        # Check visibility from each camera
        for camera_idx, camera_pose in enumerate(camera_poses):
            # Transform point to camera coordinates
            point_cam = camera_pose.rotation @ (point_3d_world - camera_pose.translation)
            
            # Check if point is in front of camera and within reasonable bounds
            if point_cam[2] > 1.0:  # Point is in front of camera
                # Project to 2D
                projected_2d = camera_model.project(point_cam.reshape(1, 3))[0]
                
                # Check if projection is within reasonable image bounds
                # Use more generous bounds since we don't know the actual image size
                if (-1000 <= projected_2d[0] <= 2000 and -1000 <= projected_2d[1] <= 1500):
                    visible_cameras.append((camera_idx, projected_2d))
            # Point is behind camera, skip
            pass
        
        # Add observations for this point from multiple cameras
        if len(visible_cameras) >= 2:  # Only add if at least 2 cameras can see it
            for camera_idx, projected_2d in visible_cameras:
                # Add Gaussian noise to observation
                noisy_point = projected_2d + np.random.normal(0, noise_std, 2)
                observations.append(Observation(camera_idx, point_idx, noisy_point))
    
    print(f"Generated {len(observations)} observations")
    
    # Create spatial intersection data
    intersection_data = SpatialIntersectionData(
        camera_poses=camera_poses,
        points_2d=observations,
        camera_model=camera_model
    )
    
    return intersection_data


def load_colmap_cameras(path_to_images_txt: str) -> Tuple[List[CameraPose], List[Observation]]:
    """
    Load camera poses and observations from COLMAP images.txt file.
    
    COLMAP images.txt format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    
    Args:
        path_to_images_txt: Path to COLMAP images.txt file
        
    Returns:
        Tuple of (camera_poses, observations) where:
        - camera_poses: List of CameraPose objects
        - observations: List of Observation objects
        
    Raises:
        FileNotFoundError: If images.txt file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path_to_images_txt)
    if not path.exists():
        raise FileNotFoundError(f"COLMAP images.txt file not found: {path_to_images_txt}")
    
    camera_poses = []
    observations = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    i = 0
    while i < len(data_lines):
        # Parse camera pose line
        pose_line = data_lines[i]
        pose_parts = pose_line.split()
        
        if len(pose_parts) < 9:
            raise ValueError(f"Invalid camera pose line: {pose_line}")
        
        # Extract camera pose data
        image_id = int(pose_parts[0])
        qw, qx, qy, qz = map(float, pose_parts[1:5])  # Quaternion
        tx, ty, tz = map(float, pose_parts[5:8])      # Translation
        camera_id = int(pose_parts[8])
        
        # Convert quaternion to rotation matrix
        quaternion = np.array([qw, qx, qy, qz])
        rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Create camera pose
        camera_pose = CameraPose(rotation_matrix, np.array([tx, ty, tz]))
        camera_poses.append(camera_pose)
        
        # Parse observations line
        if i + 1 < len(data_lines):
            obs_line = data_lines[i + 1]
            obs_parts = obs_line.split()
            
            # Observations come in groups of 3: (X, Y, POINT3D_ID)
            for j in range(0, len(obs_parts), 3):
                if j + 2 < len(obs_parts):
                    x, y = map(float, obs_parts[j:j+2])
                    point3d_id = int(obs_parts[j+2])
                    
                    # Only add observations with valid 3D points (point3d_id != -1)
                    if point3d_id != -1:
                        observation = Observation(
                            camera_index=len(camera_poses) - 1,  # 0-based index
                            point_index=point3d_id,  # Will be mapped later
                            location=np.array([x, y])
                        )
                        observations.append(observation)
        
        i += 2  # Skip to next camera (2 lines per camera)
    
    return camera_poses, observations


def load_colmap_points3D(path_to_points3D_txt: str) -> npt.NDArray[np.float64]:
    """
    Load 3D points from COLMAP points3D.txt file.
    
    COLMAP points3D.txt format:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    
    Args:
        path_to_points3D_txt: Path to COLMAP points3D.txt file
        
    Returns:
        3D points array of shape (N, 3) where N is the number of points
        
    Raises:
        FileNotFoundError: If points3D.txt file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(path_to_points3D_txt)
    if not path.exists():
        raise FileNotFoundError(f"COLMAP points3D.txt file not found: {path_to_points3D_txt}")
    
    points_3d = []
    point_id_to_index = {}  # Map COLMAP point IDs to array indices
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    
    for i, line in enumerate(data_lines):
        parts = line.split()
        
        if len(parts) < 4:
            raise ValueError(f"Invalid 3D point line: {line}")
        
        # Extract point data
        point3d_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        
        # Store point and create mapping
        points_3d.append([x, y, z])
        point_id_to_index[point3d_id] = i
    
    return np.array(points_3d, dtype=np.float64), point_id_to_index


def load_colmap_bundle(
    path_to_images_txt: str,
    path_to_points3D_txt: str,
    camera_model: CameraModel
) -> SpatialIntersectionData:
    """
    Load complete spatial intersection data from COLMAP text exports.
    
    Args:
        path_to_images_txt: Path to COLMAP images.txt file
        path_to_points3D_txt: Path to COLMAP points3D.txt file
        camera_model: CameraModel instance with intrinsics
        
    Returns:
        SpatialIntersectionData object with loaded camera poses and observations
        
    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If data is inconsistent or invalid
    """
    # Load camera poses and observations
    camera_poses, observations = load_colmap_cameras(path_to_images_txt)
    
    # Load 3D points (for reference, not used in intersection)
    points_3d, point_id_to_index = load_colmap_points3D(path_to_points3D_txt)
    
    # Validate data consistency
    if len(camera_poses) == 0:
        raise ValueError("No camera poses loaded from images.txt")
    
    if len(points_3d) == 0:
        raise ValueError("No 3D points loaded from points3D.txt")
    
    # Remap observation point indices to match the loaded 3D points
    valid_observations = []
    for obs in observations:
        if obs.point_index in point_id_to_index:
            # Update point index to match the loaded 3D points array
            new_obs = Observation(
                camera_index=obs.camera_index,
                point_index=point_id_to_index[obs.point_index],
                location=obs.location
            )
            valid_observations.append(new_obs)
        else:
            # Skip observations with invalid point IDs
            continue
    
    if len(valid_observations) == 0:
        raise ValueError("No valid observations found after point index remapping")
    
    # Create spatial intersection data
    intersection_data = SpatialIntersectionData(
        camera_poses=camera_poses,
        points_2d=valid_observations,
        camera_model=camera_model
    )
    
    return intersection_data


def validate_colmap_data(
    camera_poses: List[CameraPose],
    observations: List[Observation]
) -> Dict[str, any]:
    """
    Validate COLMAP data for consistency and quality.
    
    Args:
        camera_poses: List of camera poses
        observations: List of observations
        
    Returns:
        Dictionary with validation statistics and warnings
    """
    stats = {
        'num_cameras': len(camera_poses),
        'num_observations': len(observations),
        'warnings': [],
        'errors': []
    }
    
    # Check for valid camera poses
    for i, pose in enumerate(camera_poses):
        # Check rotation matrix orthogonality
        R = pose.rotation
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
        if orthogonality_error > 1e-6:
            stats['warnings'].append(f"Camera {i}: Rotation matrix not orthogonal (error: {orthogonality_error:.6f})")
        
        # Check rotation matrix determinant
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-6:
            stats['warnings'].append(f"Camera {i}: Rotation matrix determinant not 1 (det: {det:.6f})")
    
    # Check observation indices
    camera_indices = set(obs.camera_index for obs in observations)
    
    if max(camera_indices) >= len(camera_poses):
        stats['errors'].append(f"Invalid camera index: {max(camera_indices)} >= {len(camera_poses)}")
    
    # Check for cameras with no observations
    cameras_with_obs = set(obs.camera_index for obs in observations)
    cameras_without_obs = set(range(len(camera_poses))) - cameras_with_obs
    if cameras_without_obs:
        stats['warnings'].append(f"Cameras without observations: {cameras_without_obs}")
    
    # Compute observation statistics
    if observations:
        image_coords = np.array([obs.location for obs in observations])
        stats['mean_image_coords'] = np.mean(image_coords, axis=0)
        stats['std_image_coords'] = np.std(image_coords, axis=0)
        stats['min_image_coords'] = np.min(image_coords, axis=0)
        stats['max_image_coords'] = np.max(image_coords, axis=0)
    
    return stats


def print_colmap_summary(bundle_data: SpatialIntersectionData) -> None:
    """
    Print a summary of loaded COLMAP data.
    
    Args:
        bundle_data: SpatialIntersectionData object loaded from COLMAP
    """
    print("=" * 60)
    print("COLMAP Dataset Summary")
    print("=" * 60)
    
    print(f"Cameras: {len(bundle_data.camera_poses)}")
    print(f"Observations: {len(bundle_data.points_2d)}")
    
    # Camera statistics
    translations = np.array([pose.translation for pose in bundle_data.camera_poses])
    print(f"\nCamera Statistics:")
    print(f"  Translation range: X[{translations[:, 0].min():.2f}, {translations[:, 0].max():.2f}]")
    print(f"                    Y[{translations[:, 1].min():.2f}, {translations[:, 1].max():.2f}]")
    print(f"                    Z[{translations[:, 2].min():.2f}, {translations[:, 2].max():.2f}]")
    
    # Observation statistics
    image_coords = np.array([obs.location for obs in bundle_data.points_2d])
    print(f"\nObservation Statistics:")
    print(f"  Image coordinates: X[{image_coords[:, 0].min():.1f}, {image_coords[:, 0].max():.1f}]")
    print(f"                    Y[{image_coords[:, 1].min():.1f}, {image_coords[:, 1].max():.1f}]")
    
    # Camera model info
    print(f"\nCamera Model:")
    print(f"  Focal length: {bundle_data.camera_model.focal_length}")
    print(f"  Principal point: {bundle_data.camera_model.principal_point}")
    
    print("=" * 60)


def print_dataset_summary(data: SpatialIntersectionData) -> None:
    """
    Print a summary of the spatial intersection dataset.
    
    Args:
        data: SpatialIntersectionData object
    """
    print("=" * 60)
    print("Spatial Intersection Dataset Summary")
    print("=" * 60)
    
    print(f"Cameras: {len(data.camera_poses)}")
    print(f"Observations: {len(data.points_2d)}")
    
    # Camera statistics
    translations = np.array([pose.translation for pose in data.camera_poses])
    print(f"\nCamera Statistics:")
    print(f"  Translation range: X[{translations[:, 0].min():.2f}, {translations[:, 0].max():.2f}]")
    print(f"                    Y[{translations[:, 1].min():.2f}, {translations[:, 1].max():.2f}]")
    print(f"                    Z[{translations[:, 2].min():.2f}, {translations[:, 2].max():.2f}]")
    
    # Observation statistics
    image_coords = np.array([obs.location for obs in data.points_2d])
    print(f"\nObservation Statistics:")
    print(f"  Image coordinates: X[{image_coords[:, 0].min():.1f}, {image_coords[:, 0].max():.1f}]")
    print(f"                    Y[{image_coords[:, 1].min():.1f}, {image_coords[:, 1].max():.1f}]")
    
    # Camera model info
    print(f"\nCamera Model:")
    print(f"  Focal length: {data.camera_model.focal_length}")
    print(f"  Principal point: {data.camera_model.principal_point}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage and testing
    print("Spatial Intersection IO Utilities - Example Usage")
    print("This module provides utilities to create synthetic datasets and load COLMAP data.")
    print("To use:")
    print("1. Call create_synthetic_dataset() to generate test data")
    print("2. Use load_colmap_bundle() to load COLMAP data")
    print("3. Use print_dataset_summary() to inspect the data")
    print("4. Pass the data to spatial intersection algorithms")
    
    # Create a small example dataset
    data = create_synthetic_dataset(num_cameras=3, num_points=10, noise_std=0.5)
    print_dataset_summary(data) 