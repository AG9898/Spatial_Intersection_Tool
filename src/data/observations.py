import numpy as np
import numpy.typing as npt
from typing import List
from dataclasses import dataclass

from .camera_models import CameraModel


@dataclass
class CameraPose:
    """
    Represents a camera pose with rotation and translation.
    
    Attributes:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    """
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate the camera pose after initialization."""
        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation must be 3x3 matrix, got shape {self.rotation.shape}")
        if self.translation.shape != (3,):
            raise ValueError(f"Translation must be 3D vector, got shape {self.translation.shape}")
    
    def __repr__(self) -> str:
        return f"CameraPose(rotation_shape={self.rotation.shape}, translation_shape={self.translation.shape})"


@dataclass
class Observation:
    """
    Represents a single observation of a 3D point in an image.
    
    Attributes:
        camera_index: Index of the camera that made this observation
        point_index: Index of the 3D point being observed
        location: 2D image coordinates of the observation
    """
    camera_index: int
    point_index: int
    location: npt.NDArray[np.float64]
    
    def __post_init__(self) -> None:
        """Validate the observation after initialization."""
        if self.camera_index < 0:
            raise ValueError(f"Camera index must be non-negative, got {self.camera_index}")
        if self.point_index < 0:
            raise ValueError(f"Point index must be non-negative, got {self.point_index}")
        if self.location.shape != (2,):
            raise ValueError(f"Location must be 2D vector, got shape {self.location.shape}")
    
    def __repr__(self) -> str:
        return f"Observation(camera={self.camera_index}, point={self.point_index}, coords={self.location})"


@dataclass
class SpatialIntersectionData:
    """
    Container for all data needed for spatial intersection computation.
    
    Attributes:
        camera_poses: List of camera poses
        points_2d: List of 2D observations
        camera_model: Camera model for projection
    """
    camera_poses: List[CameraPose]
    points_2d: List[Observation]
    camera_model: CameraModel
    
    def __post_init__(self) -> None:
        """Validate the spatial intersection data after initialization."""
        if len(self.camera_poses) == 0:
            raise ValueError("At least one camera pose must be provided")
        if len(self.points_2d) == 0:
            raise ValueError("At least one observation must be provided")
        if self.camera_model is None:
            raise ValueError("Camera model must be provided")
        
        # Validate observation indices
        num_cameras = len(self.camera_poses)
        for i, obs in enumerate(self.points_2d):
            if obs.camera_index >= num_cameras:
                raise ValueError(f"Observation {i}: camera_index {obs.camera_index} out of range [0, {num_cameras-1}]")
    
    def get_observation_matrices(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        Convert observations to matrix format for optimization.
        
        Returns:
            Tuple of (camera_indices, point_indices, image_points) where:
            - camera_indices: Mx1 array of camera indices
            - point_indices: Mx1 array of point indices  
            - image_points: Mx2 array of image coordinates
        """
        num_observations = len(self.points_2d)
        
        camera_indices = np.zeros((num_observations, 1), dtype=np.int64)
        point_indices = np.zeros((num_observations, 1), dtype=np.int64)
        image_points = np.zeros((num_observations, 2), dtype=np.float64)
        
        for i, obs in enumerate(self.points_2d):
            camera_indices[i, 0] = obs.camera_index
            point_indices[i, 0] = obs.point_index
            image_points[i, :] = obs.location
        
        return camera_indices, point_indices, image_points
    
    def validate(self) -> None:
        """
        Validate the spatial intersection data for consistency.
        
        Raises:
            ValueError: If data is inconsistent or invalid
        """
        # Validate camera poses
        for i, pose in enumerate(self.camera_poses):
            if pose.rotation.shape != (3, 3):
                raise ValueError(f"Camera pose {i}: rotation must be 3x3, got {pose.rotation.shape}")
            if pose.translation.shape != (3,):
                raise ValueError(f"Camera pose {i}: translation must be (3,), got {pose.translation.shape}")
        
        # Validate observations
        num_cameras = len(self.camera_poses)
        for i, obs in enumerate(self.points_2d):
            if obs.camera_index < 0 or obs.camera_index >= num_cameras:
                raise ValueError(f"Observation {i}: camera_index {obs.camera_index} out of range [0, {num_cameras-1}]")
            if obs.location.shape != (2,):
                raise ValueError(f"Observation {i}: location must be (2,), got {obs.location.shape}")
        
        # Validate camera model
        if self.camera_model is None:
            raise ValueError("Camera model must be provided")
    
    def __repr__(self) -> str:
        return (f"SpatialIntersectionData("
                f"cameras={len(self.camera_poses)}, "
                f"observations={len(self.points_2d)})") 