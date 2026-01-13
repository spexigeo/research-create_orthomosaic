"""
Utilities for working with H3 cells and parsing image filenames.
"""

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    h3 = None

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re


def parse_image_filename(filename: str) -> Dict[str, str]:
    """
    Parse image filename to extract H3 cell, flight number, and image number.
    
    Format: {h3_cell}_{flight_number}_{image_number}.jpg
    
    Args:
        filename: Image filename (with or without path)
        
    Returns:
        Dictionary with keys: 'cell_id', 'flight_number', 'image_number', 'filename'
    """
    # Extract just the filename if path is provided
    name = Path(filename).name
    
    # Match pattern: {h3_cell}_{flight_number}_{image_number}.jpg
    pattern = r'^([a-f0-9]+)_(\d+)_(\d+)\.jpg$'
    match = re.match(pattern, name)
    
    if not match:
        raise ValueError(f"Filename '{name}' does not match expected pattern: {{h3_cell}}_{{flight_number}}_{{image_number}}.jpg")
    
    cell_id, flight_number, image_number = match.groups()
    
    return {
        'cell_id': cell_id,
        'flight_number': flight_number,
        'image_number': image_number,
        'filename': name
    }


def get_cell_images(image_dir: str, cell_id: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get all images organized by H3 cell.
    
    Args:
        image_dir: Directory containing images
        cell_id: Optional specific cell ID to filter
        
    Returns:
        Dictionary mapping cell_id to list of image filenames
    """
    image_dir = Path(image_dir)
    cell_to_images = {}
    
    for image_path in image_dir.glob('*.jpg'):
        try:
            parsed = parse_image_filename(image_path.name)
            img_cell_id = parsed['cell_id']
            
            if cell_id is None or img_cell_id == cell_id:
                if img_cell_id not in cell_to_images:
                    cell_to_images[img_cell_id] = []
                cell_to_images[img_cell_id].append(str(image_path))
        except ValueError:
            # Skip files that don't match the pattern
            continue
    
    # Sort images by flight number and image number
    for cell_id_key in cell_to_images:
        cell_to_images[cell_id_key].sort(key=lambda x: (
            parse_image_filename(x)['flight_number'],
            int(parse_image_filename(x)['image_number'])
        ))
    
    return cell_to_images


def find_central_cell(cell_ids: List[str]) -> Optional[str]:
    """
    Find the central H3 cell that is completely surrounded by 6 neighbors.
    
    In H3, each cell has exactly 6 neighbors. A central cell is one where
    all 6 neighbors are present in the cell_ids list.
    
    Args:
        cell_ids: List of H3 cell identifiers
        
    Returns:
        Cell ID of the central cell, or None if no central cell found
    """
    if not HAS_H3:
        # If h3 is not available, just return the first cell
        return cell_ids[0] if cell_ids else None
    
    cell_set = set(cell_ids)
    
    for cell_id in cell_ids:
        # Get all neighbors (should be 6 for a valid H3 cell)
        neighbors = h3.grid_ring(cell_id, k=1)
        
        # Check if all 6 neighbors are in the set
        if len(neighbors) == 6 and all(neighbor in cell_set for neighbor in neighbors):
            return cell_id
    
    return None


def get_cell_neighbors(cell_id: str, available_cells: List[str]) -> List[str]:
    """
    Get neighbors of a cell that are in the available cells list.
    
    Args:
        cell_id: H3 cell identifier
        available_cells: List of available cell IDs
        
    Returns:
        List of neighbor cell IDs that are in available_cells
    """
    if not HAS_H3:
        # If h3 is not available, return empty list
        return []
    
    available_set = set(available_cells)
    all_neighbors = h3.grid_ring(cell_id, k=1)
    return [n for n in all_neighbors if n in available_set]
