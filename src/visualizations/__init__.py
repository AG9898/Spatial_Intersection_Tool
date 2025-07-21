# Spatial Intersection Tool - Visualizations Module

from .plot_intersections import (
    plot_3d_scene,
    plot_camera_frustum,
    plot_camera_trajectory,
    plot_intersection_comparison
)
from .plot_error_histograms import (
    plot_reprojection_error_histogram,
    plot_error_vs_point_index,
    plot_error_boxplot,
    plot_cumulative_error_distribution,
    plot_error_statistics_summary
)
from .plot_summary import (
    print_intersection_summary,
    print_detailed_statistics,
    print_comparison_summary,
    print_intersection_report
)

__all__ = [
    # 3D Scene Visualization
    'plot_3d_scene',
    'plot_camera_frustum',
    'plot_camera_trajectory',
    'plot_intersection_comparison',
    
    # Error Visualization
    'plot_reprojection_error_histogram',
    'plot_error_vs_point_index',
    'plot_error_boxplot',
    'plot_cumulative_error_distribution',
    'plot_error_statistics_summary',
    
    # Summary and Reporting
    'print_intersection_summary',
    'print_detailed_statistics',
    'print_comparison_summary',
    'print_intersection_report'
] 