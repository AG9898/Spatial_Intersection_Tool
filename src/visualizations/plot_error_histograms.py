import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from ..data.observations import CameraPose


def plot_reprojection_error_histogram(
    errors: npt.NDArray[np.float64],
    title: Optional[str] = None,
    bins: int = 30,
    color: str = 'skyblue',
    alpha: float = 0.7
) -> None:
    """
    Plot histogram of reprojection errors.
    
    Args:
        errors: 1D array of reprojection errors (pixels)
        title: Optional title for the plot
        bins: Number of histogram bins
        color: Color for histogram bars
        alpha: Transparency of histogram bars
    """
    if len(errors) == 0:
        print("Warning: No errors provided for histogram")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(
        errors, 
        bins=bins, 
        color=color, 
        alpha=alpha, 
        edgecolor='black',
        linewidth=0.5
    )
    
    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    # Add vertical lines for statistics
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f} px')
    ax.axvline(median_error, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f} px')
    
    # Add text box with statistics
    stats_text = f'Mean: {mean_error:.3f} px\n'
    stats_text += f'Std: {std_error:.3f} px\n'
    stats_text += f'Median: {median_error:.3f} px\n'
    stats_text += f'Min: {min_error:.3f} px\n'
    stats_text += f'Max: {max_error:.3f} px\n'
    stats_text += f'Count: {len(errors)}'
    
    # Position text box in upper right
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Set labels and title
    ax.set_xlabel('Reprojection Error (pixels)')
    ax.set_ylabel('Frequency')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Reprojection Error Distribution')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_error_vs_point_index(
    errors: npt.NDArray[np.float64],
    point_indices: Optional[npt.NDArray[np.int64]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot reprojection errors vs point index.
    
    Args:
        errors: 1D array of reprojection errors (pixels)
        point_indices: Optional array of point indices (if None, uses sequential indices)
        title: Optional title for the plot
    """
    if len(errors) == 0:
        print("Warning: No errors provided for plot")
        return
    
    if point_indices is None:
        point_indices = np.arange(len(errors))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot errors vs point index
    ax.scatter(point_indices, errors, alpha=0.6, s=30, color='blue')
    
    # Add mean line
    mean_error = np.mean(errors)
    ax.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f} px')
    
    # Add std bands
    std_error = np.std(errors)
    ax.axhline(mean_error + std_error, color='orange', linestyle=':', linewidth=1, label=f'Mean + Std: {mean_error + std_error:.3f} px')
    ax.axhline(mean_error - std_error, color='orange', linestyle=':', linewidth=1, label=f'Mean - Std: {mean_error - std_error:.3f} px')
    
    # Set labels and title
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Reprojection Error (pixels)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Reprojection Error vs Point Index')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_error_boxplot(
    errors_by_camera: list[npt.NDArray[np.float64]],
    camera_names: Optional[list[str]] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot boxplot of reprojection errors by camera.
    
    Args:
        errors_by_camera: List of error arrays, one per camera
        camera_names: Optional list of camera names
        title: Optional title for the plot
    """
    if not errors_by_camera:
        print("Warning: No error data provided for boxplot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    bp = ax.boxplot(errors_by_camera, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(errors_by_camera)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis labels
    if camera_names is None:
        camera_names = [f'Camera {i}' for i in range(len(errors_by_camera))]
    
    ax.set_xticklabels(camera_names, rotation=45)
    
    # Set labels and title
    ax.set_xlabel('Camera')
    ax.set_ylabel('Reprojection Error (pixels)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Reprojection Error Distribution by Camera')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_cumulative_error_distribution(
    errors: npt.NDArray[np.float64],
    title: Optional[str] = None,
    color: str = 'blue'
) -> None:
    """
    Plot cumulative distribution of reprojection errors.
    
    Args:
        errors: 1D array of reprojection errors (pixels)
        title: Optional title for the plot
        color: Color for the plot line
    """
    if len(errors) == 0:
        print("Warning: No errors provided for cumulative distribution")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort errors
    sorted_errors = np.sort(errors)
    
    # Calculate cumulative distribution
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Plot cumulative distribution
    ax.plot(sorted_errors, cumulative, color=color, linewidth=2)
    
    # Add reference lines
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50%')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95%')
    
    # Find percentiles
    p50 = np.percentile(errors, 50)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    
    # Add vertical lines for percentiles
    ax.axvline(p50, color='red', linestyle=':', alpha=0.7)
    ax.axvline(p90, color='orange', linestyle=':', alpha=0.7)
    ax.axvline(p95, color='green', linestyle=':', alpha=0.7)
    
    # Add text annotations
    ax.text(p50, 0.5, f' 50%: {p50:.3f} px', verticalalignment='center')
    ax.text(p90, 0.9, f' 90%: {p90:.3f} px', verticalalignment='center')
    ax.text(p95, 0.95, f' 95%: {p95:.3f} px', verticalalignment='center')
    
    # Set labels and title
    ax.set_xlabel('Reprojection Error (pixels)')
    ax.set_ylabel('Cumulative Probability')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Cumulative Error Distribution')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_error_statistics_summary(
    errors: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Create a comprehensive error statistics summary plot.
    
    Args:
        errors: 1D array of reprojection errors (pixels)
        title: Optional title for the plot
    """
    if len(errors) == 0:
        print("Warning: No errors provided for statistics summary")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Histogram
    ax1.hist(errors, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    ax1.set_xlabel('Reprojection Error (pixels)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    ax2.boxplot(errors, patch_artist=True)
    ax2.set_ylabel('Reprojection Error (pixels)')
    ax2.set_title('Error Statistics')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Error vs index
    ax2_alt = ax2.twinx()
    ax2_alt.scatter(np.arange(len(errors)), errors, alpha=0.5, s=10, color='blue')
    ax2_alt.set_ylabel('Error (pixels)')
    
    # Subplot 3: Cumulative distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax3.plot(sorted_errors, cumulative, color='green', linewidth=2)
    ax3.set_xlabel('Reprojection Error (pixels)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Statistics text
    ax4.axis('off')
    stats_text = f'Error Statistics Summary\n\n'
    stats_text += f'Count: {len(errors)}\n'
    stats_text += f'Mean: {np.mean(errors):.3f} px\n'
    stats_text += f'Std: {np.std(errors):.3f} px\n'
    stats_text += f'Median: {np.median(errors):.3f} px\n'
    stats_text += f'Min: {np.min(errors):.3f} px\n'
    stats_text += f'Max: {np.max(errors):.3f} px\n\n'
    stats_text += f'Percentiles:\n'
    stats_text += f'  25%: {np.percentile(errors, 25):.3f} px\n'
    stats_text += f'  50%: {np.percentile(errors, 50):.3f} px\n'
    stats_text += f'  75%: {np.percentile(errors, 75):.3f} px\n'
    stats_text += f'  90%: {np.percentile(errors, 90):.3f} px\n'
    stats_text += f'  95%: {np.percentile(errors, 95):.3f} px\n'
    stats_text += f'  99%: {np.percentile(errors, 99):.3f} px'
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Reprojection Error Statistics Summary', fontsize=16)
    
    plt.tight_layout()
    plt.show() 