"""
Detect scanlines from camera poses by analyzing which images lie on straight lines.
"""
import json
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename."""
    match = re.search(r'_(\d{4})\.jpg$', image_name)
    return int(match.group(1)) if match else -1


def calculate_heading(lat1, lon1, lat2, lon2):
    """Calculate heading (bearing) between two GPS points in degrees."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    heading = np.degrees(np.arctan2(y, x))
    return (heading + 360) % 360  # Normalize to 0-360


def detect_scanlines(poses_file: str = "outputs/camera_poses.json") -> Dict:
    """
    Detect scanlines by analyzing camera positions and headings.
    
    Returns:
        Dictionary with scanline assignments: {scanline_id: [image_numbers]}
    """
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    # Filter to poses with GPS
    poses_with_gps = [p for p in poses if p.get('gps') is not None]
    
    # Sort by image number
    poses_with_gps.sort(key=lambda p: parse_image_number(p['image_name']))
    
    # Extract data
    image_numbers = [parse_image_number(p['image_name']) for p in poses_with_gps]
    lats = np.array([p['gps']['latitude'] for p in poses_with_gps])
    lons = np.array([p['gps']['longitude'] for p in poses_with_gps])
    
    # Calculate headings between consecutive images
    headings = []
    distances = []
    for i in range(len(lats) - 1):
        heading = calculate_heading(lats[i], lons[i], lats[i+1], lons[i+1])
        headings.append(heading)
        
        # Distance in meters
        lat_diff = (lats[i+1] - lats[i]) * 111320
        lon_diff = (lons[i+1] - lons[i]) * 111320 * np.cos(np.radians(lats[i]))
        dist = np.sqrt(lat_diff**2 + lon_diff**2)
        distances.append(dist)
    
    # Detect scanline boundaries by finding large heading changes
    # A scanline turn typically involves a heading change > 90 degrees
    heading_changes = []
    for i in range(len(headings) - 1):
        change = abs(headings[i+1] - headings[i])
        if change > 180:
            change = 360 - change
        heading_changes.append(change)
    
    # Find scanline boundaries (large heading changes)
    # Use a higher threshold and require sustained direction changes
    threshold = 120  # degrees - more conservative
    boundaries = [0]  # Start with first image
    
    # Look for sustained direction changes (not just single spikes)
    for i in range(len(heading_changes) - 2):
        # Check if this is a sustained turn (current and next change are both large)
        if heading_changes[i] > threshold and heading_changes[i+1] > threshold:
            boundaries.append(i + 2)  # Image after the turn completes
        # Or if it's a very large single turn (180 degrees = U-turn)
        elif heading_changes[i] > 170:
            boundaries.append(i + 1)
    
    boundaries.append(len(image_numbers))  # End with last image
    
    # Remove duplicate boundaries
    boundaries = sorted(list(set(boundaries)))
    
    # Group images into scanlines
    scanlines = {}
    for scanline_id, start_idx in enumerate(boundaries[:-1], 1):
        end_idx = boundaries[scanline_id] if scanline_id < len(boundaries) - 1 else len(image_numbers)
        scanline_images = image_numbers[start_idx:end_idx]
        scanlines[scanline_id] = scanline_images
    
    return scanlines


def verify_scanline_straightness(poses_file: str = "outputs/camera_poses.json", 
                                 scanlines: Dict = None) -> Dict:
    """
    Verify that images in each scanline lie approximately on a straight line.
    
    Returns:
        Dictionary with scanline quality metrics
    """
    if scanlines is None:
        scanlines = detect_scanlines(poses_file)
    
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    poses_dict = {parse_image_number(p['image_name']): p for p in poses if p.get('gps')}
    
    results = {}
    for scanline_id, image_nums in scanlines.items():
        if len(image_nums) < 2:
            continue
        
        # Get positions for this scanline
        positions = []
        for img_num in image_nums:
            if img_num in poses_dict:
                p = poses_dict[img_num]
                positions.append([p['gps']['latitude'], p['gps']['longitude']])
        
        if len(positions) < 2:
            continue
        
        positions = np.array(positions)
        
        # Fit a line to the positions (using PCA to find principal direction)
        # Center the data
        center = positions.mean(axis=0)
        centered = positions - center
        
        # PCA to find principal direction
        if len(centered) > 1:
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_dir = eigenvecs[:, np.argmax(eigenvals)]
            
            # Project points onto the line
            projections = np.dot(centered, principal_dir)
            projected_points = center + np.outer(projections, principal_dir)
            
            # Calculate RMS distance from line
            distances = np.linalg.norm(positions - projected_points, axis=1)
            rms_error = np.sqrt(np.mean(distances**2))
            
            # Convert to meters (rough approximation)
            rms_error_m = rms_error * 111320
            
            results[scanline_id] = {
                'image_numbers': image_nums,
                'num_images': len(image_nums),
                'rms_error_meters': rms_error_m,
                'start_image': min(image_nums),
                'end_image': max(image_nums)
            }
    
    return results


if __name__ == "__main__":
    print("Detecting scanlines from camera poses...")
    scanlines = detect_scanlines()
    
    print(f"\nDetected {len(scanlines)} scanlines:")
    for scanline_id, image_nums in sorted(scanlines.items()):
        print(f"  Scanline {scanline_id}: {min(image_nums):04d}-{max(image_nums):04d} ({len(image_nums)} images)")
    
    print("\nVerifying scanline straightness...")
    verification = verify_scanline_straightness(scanlines=scanlines)
    
    print("\nScanline Quality Metrics:")
    for scanline_id, metrics in sorted(verification.items()):
        print(f"  Scanline {scanline_id}: {metrics['num_images']} images, "
              f"RMS error: {metrics['rms_error_meters']:.2f} m, "
              f"Images: {metrics['start_image']:04d}-{metrics['end_image']:04d}")
    
    # Save scanlines
    output_file = "outputs/scanlines.json"
    scanlines_data = {
        'scanlines': {str(k): v for k, v in scanlines.items()},
        'verification': {str(k): v for k, v in verification.items()}
    }
    with open(output_file, 'w') as f:
        json.dump(scanlines_data, f, indent=2)
    print(f"\nSaved scanlines to: {output_file}")
