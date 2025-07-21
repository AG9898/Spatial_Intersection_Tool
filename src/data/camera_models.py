import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple


class CameraModel:
    """
    Simple pinhole camera model for spatial intersection.
    
    Implements perspective projection with focal length and principal point offset.
    Optional radial distortion support (placeholder for future implementation).
    """
    
    def __init__(
        self,
        focal_length: float,
        principal_point: Tuple[float, float],
        distortion_coeffs: Optional[list[float]] = None
    ) -> None:
        """
        Initialize camera model.
        
        Args:
            focal_length: Camera focal length in pixels
            principal_point: Principal point (cx, cy) in pixels
            distortion_coeffs: Optional radial distortion coefficients (unused for now)
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.distortion_coeffs = distortion_coeffs
    
    def project(
        self, 
        points_3d: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Project 3D points to 2D image coordinates using pinhole model.
        
        Args:
            points_3d: 3D points in camera coordinates (Nx3 array)
            
        Returns:
            2D points in image coordinates (Nx2 array)
        """
        # Validate input shape
        assert points_3d.ndim == 2 and points_3d.shape[1] == 3, \
            f"points_3d must be Nx3 array, got shape {points_3d.shape}"
        
        # Perspective projection: x' = X/Z, y' = Y/Z
        # Avoid division by zero
        z_coords = points_3d[:, 2]
        valid_mask = z_coords > 0
        
        points_2d = np.zeros((points_3d.shape[0], 2))
        points_2d[valid_mask] = points_3d[valid_mask, :2] / z_coords[valid_mask, np.newaxis]
        
        # Apply focal length and principal point offset
        points_2d[valid_mask, 0] = points_2d[valid_mask, 0] * self.focal_length + self.principal_point[0]
        points_2d[valid_mask, 1] = points_2d[valid_mask, 1] * self.focal_length + self.principal_point[1]
        
        # TODO: Apply distortion if coefficients are provided
        # if self.distortion_coeffs is not None:
        #     points_2d = self._apply_distortion(points_2d)
        
        return points_2d
    
    def __repr__(self) -> str:
        return f"CameraModel(focal_length={self.focal_length}, principal_point={self.principal_point})" 