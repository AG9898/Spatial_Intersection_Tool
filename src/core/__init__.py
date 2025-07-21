# Spatial Intersection Tool - Core Module

from .geometry_utils import (
    ray_from_camera,
    compute_ray_distance,
    compute_ray_intersection_point,
    validate_ray_geometry
)
from .spatial_intersection import (
    triangulate_point,
    run_spatial_intersection,
    compute_triangulation_quality,
    print_triangulation_summary
)

__all__ = [
    # Geometry utilities
    'ray_from_camera',
    'compute_ray_distance',
    'compute_ray_intersection_point',
    'validate_ray_geometry',
    
    # Spatial intersection
    'triangulate_point',
    'run_spatial_intersection',
    'compute_triangulation_quality',
    'print_triangulation_summary'
] 