"""
Visualize camera poses from extracted GPS coordinates.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import re


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename (e.g., 0001 from ..._0001.jpg)."""
    match = re.search(r'_(\d{4})\.jpg$', image_name)
    if match:
        return int(match.group(1))
    return -1


def visualize_camera_poses(poses_file: str = "outputs/camera_poses.json",
                           output_file: str = "test_visualization/camera_poses.png"):
    """
    Visualize camera positions and orientations.
    
    Args:
        poses_file: Path to JSON file with camera poses
        output_file: Path to save visualization
    """
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    # Filter to only poses with GPS
    poses_with_gps = [p for p in poses if p.get('gps') is not None]
    
    if not poses_with_gps:
        print("No GPS data available for visualization")
        return
    
    print(f"Visualizing {len(poses_with_gps)} camera poses")
    
    # Extract coordinates
    lats = [p['gps']['latitude'] for p in poses_with_gps]
    lons = [p['gps']['longitude'] for p in poses_with_gps]
    alts = [p['gps']['altitude'] for p in poses_with_gps]
    image_names = [p['image_name'] for p in poses_with_gps]
    image_numbers = [parse_image_number(name) for name in image_names]
    
    # Sort by image number
    sorted_indices = np.argsort(image_numbers)
    lats = np.array(lats)[sorted_indices]
    lons = np.array(lons)[sorted_indices]
    alts = np.array(alts)[sorted_indices]
    image_names = np.array(image_names)[sorted_indices]
    image_numbers = np.array(image_numbers)[sorted_indices]
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Top view (lat/lon)
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(lons, lats, c=image_numbers, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Draw lines connecting sequential images
    for i in range(len(lons) - 1):
        if image_numbers[i+1] == image_numbers[i] + 1:
            ax1.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], 
                    'r-', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Camera Positions (Top View)\nColored by Image Number')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    plt.colorbar(scatter, ax=ax1, label='Image Number')
    
    # Side view (lon/altitude)
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(lons, alts, c=image_numbers, cmap='viridis', 
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    for i in range(len(lons) - 1):
        if image_numbers[i+1] == image_numbers[i] + 1:
            ax2.plot([lons[i], lons[i+1]], [alts[i], alts[i+1]], 
                    'r-', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Camera Positions (Side View - Longitude)')
    ax2.grid(True, alpha=0.3)
    
    # Side view (lat/altitude)
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(lats, alts, c=image_numbers, cmap='viridis', 
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    for i in range(len(lats) - 1):
        if image_numbers[i+1] == image_numbers[i] + 1:
            ax3.plot([lats[i], lats[i+1]], [alts[i], alts[i+1]], 
                    'r-', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Camera Positions (Side View - Latitude)')
    ax3.grid(True, alpha=0.3)
    
    # 3D view
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    scatter3d = ax4.scatter(lons, lats, alts, c=image_numbers, cmap='viridis',
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    for i in range(len(lons) - 1):
        if image_numbers[i+1] == image_numbers[i] + 1:
            ax4.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], 
                    [alts[i], alts[i+1]], 'r-', alpha=0.3, linewidth=1)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_zlabel('Altitude (m)')
    ax4.set_title('Camera Positions (3D View)')
    
    plt.tight_layout()
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved camera pose visualization to: {output_file}")
    
    # Print statistics
    print(f"\nFlight Statistics:")
    print(f"  Total images: {len(poses_with_gps)}")
    print(f"  Latitude range: {min(lats):.6f} to {max(lats):.6f} ({max(lats)-min(lats):.6f} degrees)")
    print(f"  Longitude range: {min(lons):.6f} to {max(lons):.6f} ({max(lons)-min(lons):.6f} degrees)")
    print(f"  Altitude range: {min(alts):.1f} to {max(alts):.1f} m ({max(alts)-min(alts):.1f} m)")
    print(f"  Image number range: {min(image_numbers)} to {max(image_numbers)}")
    
    # Try to detect scanlines
    print(f"\nAnalyzing flight pattern...")
    # Calculate distances between consecutive images
    distances = []
    for i in range(len(lons) - 1):
        if image_numbers[i+1] == image_numbers[i] + 1:
            # Approximate distance in meters (rough calculation)
            lat_diff = (lats[i+1] - lats[i]) * 111320  # meters per degree latitude
            lon_diff = (lons[i+1] - lons[i]) * 111320 * np.cos(np.radians(lats[i]))  # meters per degree longitude
            dist = np.sqrt(lat_diff**2 + lon_diff**2)
            distances.append(dist)
    
    if distances:
        print(f"  Average distance between consecutive images: {np.mean(distances):.1f} m")
        print(f"  Min distance: {np.min(distances):.1f} m")
        print(f"  Max distance: {np.max(distances):.1f} m")


if __name__ == "__main__":
    visualize_camera_poses()
