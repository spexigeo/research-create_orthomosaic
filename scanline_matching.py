"""
Use scanline information to improve feature matching.
"""
import json
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename."""
    match = re.search(r'_(\d{4})\.jpg$', image_name)
    return int(match.group(1)) if match else -1


class ScanlineMatcher:
    """Manages scanline information and provides matching priorities."""
    
    def __init__(self, scanlines: Dict[int, List[int]]):
        """
        Initialize with scanline assignments.
        
        Args:
            scanlines: Dictionary mapping scanline_id to list of image numbers
        """
        self.scanlines = scanlines
        
        # Build reverse mapping: image_number -> scanline_id
        self.image_to_scanline = {}
        for scanline_id, image_nums in scanlines.items():
            for img_num in image_nums:
                self.image_to_scanline[img_num] = scanline_id
        
        # Build scanline adjacency (scanlines that are next to each other)
        self.adjacent_scanlines = self._build_adjacency()
    
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """Build adjacency map of which scanlines are next to each other."""
        adjacent = defaultdict(set)
        scanline_ids = sorted(self.scanlines.keys())
        
        for i, scanline_id in enumerate(scanline_ids):
            # Add previous scanline
            if i > 0:
                adjacent[scanline_id].add(scanline_ids[i-1])
            # Add next scanline
            if i < len(scanline_ids) - 1:
                adjacent[scanline_id].add(scanline_ids[i+1])
        
        return dict(adjacent)
    
    def get_scanline(self, image_number: int) -> int:
        """Get scanline ID for an image number."""
        return self.image_to_scanline.get(image_number, -1)
    
    def get_matching_priority(self, img1_num: int, img2_num: int) -> int:
        """
        Get matching priority for two images.
        Lower number = higher priority.
        
        Returns:
            Priority level:
            0: Same scanline, adjacent images (highest priority)
            1: Same scanline, within 2 images
            2: Same scanline, within 5 images
            3: Adjacent scanlines, similar positions
            4: Adjacent scanlines, any position
            5: Same scanline, far apart
            10: Different scanlines, not adjacent (lowest priority)
        """
        scanline1 = self.get_scanline(img1_num)
        scanline2 = self.get_scanline(img2_num)
        
        if scanline1 == -1 or scanline2 == -1:
            return 10  # Unknown scanline
        
        if scanline1 == scanline2:
            # Same scanline - prioritize by distance
            distance = abs(img1_num - img2_num)
            if distance == 1:
                return 0  # Adjacent images
            elif distance <= 2:
                return 1
            elif distance <= 5:
                return 2
            else:
                return 5  # Same scanline but far apart
        else:
            # Different scanlines
            if scanline2 in self.adjacent_scanlines.get(scanline1, set()):
                # Adjacent scanlines - check if positions are similar
                scanline1_images = sorted(self.scanlines[scanline1])
                scanline2_images = sorted(self.scanlines[scanline2])
                
                # Find position within scanline
                pos1 = scanline1_images.index(img1_num) if img1_num in scanline1_images else -1
                pos2 = scanline2_images.index(img2_num) if img2_num in scanline2_images else -1
                
                if pos1 >= 0 and pos2 >= 0:
                    # Normalize positions (0-1 scale)
                    norm_pos1 = pos1 / max(len(scanline1_images) - 1, 1)
                    norm_pos2 = pos2 / max(len(scanline2_images) - 1, 1)
                    
                    # If positions are similar, higher priority
                    if abs(norm_pos1 - norm_pos2) < 0.3:  # Within 30% of scanline length
                        return 3
                
                return 4  # Adjacent scanlines but different positions
            else:
                return 10  # Not adjacent scanlines
    
    def get_matching_radius(self, image_number: int) -> Dict[str, int]:
        """
        Get matching radius for an image.
        
        Returns:
            Dictionary with:
            - same_scanline_max_distance: Max distance on same scanline
            - adjacent_scanlines: Whether to match with adjacent scanlines
            - max_scanline_distance: Max scanline distance (0 = same, 1 = adjacent, etc.)
        """
        scanline = self.get_scanline(image_number)
        
        if scanline == -1:
            return {
                'same_scanline_max_distance': 10,
                'adjacent_scanlines': True,
                'max_scanline_distance': 1
            }
        
        # For same scanline, match with neighbors within reasonable distance
        # For adjacent scanlines, match with similar positions
        return {
            'same_scanline_max_distance': 5,  # Match with up to 5 images away on same scanline
            'adjacent_scanlines': True,
            'max_scanline_distance': 1  # Only match with adjacent scanlines
        }
    
    def should_match(self, img1_num: int, img2_num: int) -> bool:
        """Determine if two images should be matched based on scanline proximity."""
        priority = self.get_matching_priority(img1_num, img2_num)
        return priority < 10  # Only match if priority is reasonable
    
    def get_priority_pairs(self, image_numbers: List[int]) -> List[Tuple[int, int, int]]:
        """
        Get all image pairs sorted by matching priority.
        
        Returns:
            List of (img1_num, img2_num, priority) tuples, sorted by priority
        """
        pairs = []
        for i, img1 in enumerate(image_numbers):
            for img2 in image_numbers[i+1:]:
                priority = self.get_matching_priority(img1, img2)
                pairs.append((img1, img2, priority))
        
        # Sort by priority (lower = better)
        pairs.sort(key=lambda x: x[2])
        return pairs


def load_scanlines_from_analysis(analysis_file: str = "outputs/scanlines_analysis.json") -> Dict[int, List[int]]:
    """Load scanlines from analysis file (using user-provided scanlines)."""
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    # Use user scanlines
    user_scanlines = data.get('user_scanlines', {})
    
    scanlines = {}
    for scanline_id_str, metrics in user_scanlines.items():
        scanline_id = int(scanline_id_str)
        image_nums = metrics.get('image_numbers', [])
        if image_nums:
            scanlines[scanline_id] = image_nums
    
    return scanlines


if __name__ == "__main__":
    # Test the scanline matcher
    scanlines = load_scanlines_from_analysis()
    
    print(f"Loaded {len(scanlines)} scanlines")
    for scanline_id, image_nums in sorted(scanlines.items()):
        print(f"  Scanline {scanline_id}: {min(image_nums):04d}-{max(image_nums):04d} ({len(image_nums)} images)")
    
    matcher = ScanlineMatcher(scanlines)
    
    # Test with image 0040 (should be on scanline 3)
    test_image = 40
    print(f"\nTesting matching priorities for image {test_image:04d}:")
    print(f"  Scanline: {matcher.get_scanline(test_image)}")
    
    # Test with various other images
    test_pairs = [
        (40, 39),  # Same scanline, adjacent
        (40, 41),  # Same scanline, adjacent
        (40, 38),  # Same scanline, 2 away
        (40, 15),  # Different scanline (scanline 2)
        (40, 50),  # Different scanline (scanline 4)
        (40, 100), # Different scanline, far
    ]
    
    print("\nMatching priorities:")
    for img1, img2 in test_pairs:
        priority = matcher.get_matching_priority(img1, img2)
        should_match = matcher.should_match(img1, img2)
        print(f"  Image {img1:04d} <-> {img2:04d}: Priority {priority}, Should match: {should_match}")
