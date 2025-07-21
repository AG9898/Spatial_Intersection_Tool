import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..data.observations import CameraPose


def plot_3d_scene(
    camera_poses: List[CameraPose],
    points_3d: npt.NDArray[np.float64],
    rays: Optional[List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = None,
    title: Optional[str] = None,
    camera_scale: float = 1.0,
    point_size: float = 20.0,
    ray_alpha: float = 0.3,
    show_camera_frustums: bool = True
) -> None:
    """
    Plot 3D scene with cameras, points, and optional rays.
    
    Args:
        camera_poses: List of camera poses
        points_3d: 3D points array (Nx3)
        rays: Optional list of (origin, direction) tuples for ray visualization
        title: Optional title for the plot
        camera_scale: Scale factor for camera frustums
        point_size: Size of 3D points in scatter plot
        ray_alpha: Transparency for ray lines
        show_camera_frustums: Whether to show camera frustums or just points
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if points_3d.shape[0] > 0:
        ax.scatter(
            points_3d[:, 0], 
            points_3d[:, 1], 
            points_3d[:, 2],
            c='blue', 
            s=point_size, 
            alpha=0.7, 
            label=f'3D Points ({points_3d.shape[0]})'
        )
    
    # Plot cameras
    camera_positions = np.array([pose.translation for pose in camera_poses])
    
    if show_camera_frustums:
        # Plot camera frustums
        for i, pose in enumerate(camera_poses):
            plot_camera_frustum(ax, pose, scale=camera_scale, color=f'C{i}')
    else:
        # Plot cameras as points
        ax.scatter(
            camera_positions[:, 0],
            camera_positions[:, 1], 
            camera_positions[:, 2],
            c='red',
            s=100,
            marker='^',
            label=f'Cameras ({len(camera_poses)})'
        )
    
    # Plot rays if provided
    if rays is not None:
        for origin, direction in rays:
            # Plot ray as a line segment
            end_point = origin + direction * 10.0  # Extend ray by 10 units
            ax.plot(
                [origin[0], end_point[0]],
                [origin[1], end_point[1]], 
                [origin[2], end_point[2]],
                'g-', 
                alpha=ray_alpha, 
                linewidth=1
            )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('3D Scene Visualization')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    # Auto-scale to fit all data
    if points_3d.shape[0] > 0:
        all_points = np.vstack([points_3d, camera_positions])
    else:
        all_points = camera_positions
    
    if all_points.shape[0] > 0:
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        
        # Add some padding
        padding = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.1
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_zlim(z_min - padding, z_max + padding)
    
    plt.tight_layout()
    plt.show()


def plot_camera_frustum(
    ax: Axes3D,
    pose: CameraPose,
    scale: float = 1.0,
    color: str = 'red'
) -> None:
    """
    Plot a camera frustum in 3D.
    
    Args:
        ax: 3D matplotlib axis
        pose: Camera pose
        scale: Scale factor for frustum size
        color: Color for the frustum
    """
    # Define frustum vertices in camera coordinates
    # Simple pyramid frustum
    near = 0.5 * scale
    far = 2.0 * scale
    width = 1.0 * scale
    
    # Frustum vertices in camera coordinates
    vertices_cam = np.array([
        # Near plane
        [-width, -width, near],
        [width, -width, near],
        [width, width, near],
        [-width, width, near],
        # Far plane
        [-width/2, -width/2, far],
        [width/2, -width/2, far],
        [width/2, width/2, far],
        [-width/2, width/2, far]
    ])
    
    # Transform to world coordinates
    vertices_world = pose.rotation @ vertices_cam.T + pose.translation.reshape(3, 1)
    vertices_world = vertices_world.T
    
    # Define faces (triangles and quads)
    faces = [
        # Near plane
        [vertices_world[0], vertices_world[1], vertices_world[2], vertices_world[3]],
        # Far plane
        [vertices_world[4], vertices_world[5], vertices_world[6], vertices_world[7]],
        # Side faces
        [vertices_world[0], vertices_world[1], vertices_world[5], vertices_world[4]],
        [vertices_world[1], vertices_world[2], vertices_world[6], vertices_world[5]],
        [vertices_world[2], vertices_world[3], vertices_world[7], vertices_world[6]],
        [vertices_world[3], vertices_world[0], vertices_world[4], vertices_world[7]]
    ]
    
    # Create poly3d collection
    poly3d = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='black')
    ax.add_collection3d(poly3d)
    
    # Plot camera center
    ax.scatter(
        pose.translation[0], 
        pose.translation[1], 
        pose.translation[2],
        c=color, 
        s=50, 
        marker='o'
    )


def plot_camera_trajectory(
    camera_poses: List[CameraPose],
    title: Optional[str] = None,
    show_orientations: bool = True
) -> None:
    """
    Plot camera trajectory in 3D.
    
    Args:
        camera_poses: List of camera poses
        title: Optional title for the plot
        show_orientations: Whether to show camera orientations
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions
    positions = np.array([pose.translation for pose in camera_poses])
    
    # Plot trajectory
    ax.plot(
        positions[:, 0], 
        positions[:, 1], 
        positions[:, 2],
        'b-o', 
        linewidth=2, 
        markersize=6,
        label='Camera Trajectory'
    )
    
    # Plot camera orientations if requested
    if show_orientations:
        for i, pose in enumerate(camera_poses):
            # Plot camera coordinate frame
            origin = pose.translation
            scale = 2.0
            
            # X axis (red)
            x_end = origin + pose.rotation[:, 0] * scale
            ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 'r-', linewidth=2)
            
            # Y axis (green)
            y_end = origin + pose.rotation[:, 1] * scale
            ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 'g-', linewidth=2)
            
            # Z axis (blue)
            z_end = origin + pose.rotation[:, 2] * scale
            ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 'b-', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Camera Trajectory')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_intersection_comparison(
    points_3d_true: npt.NDArray[np.float64],
    points_3d_estimated: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Plot comparison between true and estimated 3D points.
    
    Args:
        points_3d_true: True 3D points (Nx3)
        points_3d_estimated: Estimated 3D points (Nx3)
        title: Optional title for the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true points
    ax.scatter(
        points_3d_true[:, 0], 
        points_3d_true[:, 1], 
        points_3d_true[:, 2],
        c='green', 
        s=50, 
        alpha=0.7, 
        label='True Points',
        marker='o'
    )
    
    # Plot estimated points
    ax.scatter(
        points_3d_estimated[:, 0], 
        points_3d_estimated[:, 1], 
        points_3d_estimated[:, 2],
        c='red', 
        s=50, 
        alpha=0.7, 
        label='Estimated Points',
        marker='^'
    )
    
    # Plot connections between true and estimated points
    for i in range(min(len(points_3d_true), len(points_3d_estimated))):
        ax.plot(
            [points_3d_true[i, 0], points_3d_estimated[i, 0]],
            [points_3d_true[i, 1], points_3d_estimated[i, 1]],
            [points_3d_true[i, 2], points_3d_estimated[i, 2]],
            'k-', 
            alpha=0.3, 
            linewidth=1
        )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('True vs Estimated 3D Points')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.show() 