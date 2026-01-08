"""
Export point cloud to PLY format for visualization in MeshLab.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def export_point_cloud_to_ply(
    point_cloud_json: str,
    output_ply: str,
    color_by_error: bool = True
):
    """
    Export point cloud from JSON to PLY format.
    
    Args:
        point_cloud_json: Path to JSON file with point cloud data
        output_ply: Path to output PLY file
        color_by_error: If True, color points by reprojection error
    """
    # Load point cloud
    with open(point_cloud_json, 'r') as f:
        pc_data = json.load(f)
    
    points = pc_data['points']
    
    if not points:
        raise ValueError("Point cloud is empty")
    
    # Extract 3D coordinates
    xyz = np.array([p['point_3d'] for p in points])
    
    # Get reprojection errors for coloring
    if color_by_error:
        errors = np.array([p['reprojection_error_mean'] for p in points])
        # Normalize errors to [0, 1] for color mapping
        if errors.max() > errors.min():
            errors_norm = (errors - errors.min()) / (errors.max() - errors.min())
        else:
            errors_norm = np.zeros_like(errors)
        
        # Color map: blue (low error) to red (high error)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = (errors_norm * 255).astype(np.uint8)  # Red channel
        colors[:, 2] = ((1 - errors_norm) * 255).astype(np.uint8)  # Blue channel
    else:
        # Default color (white)
        colors = np.ones((len(points), 3), dtype=np.uint8) * 255
    
    # Write PLY file
    output_path = Path(output_ply)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for i in range(len(points)):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"Exported {len(points)} points to PLY file: {output_ply}")
    print(f"  X range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"  Y range: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    print(f"  Z range: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
    
    if color_by_error:
        print(f"  Colors: Blue (low error) to Red (high error)")
        print(f"  Error range: [{errors.min():.3f}, {errors.max():.3f}] pixels")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ply_export.py <point_cloud_json> [output_ply]")
        print("Example: python ply_export.py outputs/point_cloud_cell_8928d89ac57ffff.json outputs/point_cloud.ply")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_ply = sys.argv[2] if len(sys.argv) > 2 else input_json.replace('.json', '.ply')
    
    export_point_cloud_to_ply(input_json, output_ply)
