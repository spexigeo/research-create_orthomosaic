"""
Visualize how many images overlap with each image (after 10% threshold).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_overlap_counts(
    overlaps_file: str = "outputs/footprint_overlaps.json",
    min_overlap_threshold: float = 50.0,
    output_file: str = "test_visualization/overlap_counts_per_image.png"
):
    """
    Create a visualization showing how many images overlap with each image.
    
    Args:
        overlaps_file: Path to footprint_overlaps.json
        min_overlap_threshold: Minimum overlap percentage threshold
        output_file: Path to output PNG file
    """
    # Load overlaps
    with open(overlaps_file, 'r') as f:
        data = json.load(f)
    
    # Count overlaps per image
    overlap_counts = {}
    
    for overlap in data['overlaps']:
        if overlap['overlap_percentage'] >= min_overlap_threshold:
            img1 = overlap['image1']
            img2 = overlap['image2']
            
            # Count overlaps for each image
            overlap_counts[img1] = overlap_counts.get(img1, 0) + 1
            overlap_counts[img2] = overlap_counts.get(img2, 0) + 1
    
    # Extract image numbers for sorting
    def get_image_number(img_name):
        try:
            # Extract number from filename like "8928d89ac57ffff_172550_0001.jpg"
            parts = img_name.split('_')
            if len(parts) >= 3:
                num_str = parts[-1].replace('.jpg', '')
                return int(num_str)
        except:
            return 0
        return 0
    
    # Sort images by number
    sorted_images = sorted(overlap_counts.keys(), key=get_image_number)
    
    # Get counts in order
    counts = [overlap_counts.get(img, 0) for img in sorted_images]
    image_numbers = [get_image_number(img) for img in sorted_images]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Bar chart
    ax1.bar(image_numbers, counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Image Number', fontsize=12)
    ax1.set_ylabel(f'Number of Overlapping Images (>= {min_overlap_threshold}% overlap)', fontsize=12)
    ax1.set_title(f'Number of Overlapping Images per Image (Threshold: {min_overlap_threshold}%)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, max(image_numbers) + 1)
    
    # Add statistics text
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)
    
    stats_text = f'Mean: {mean_count:.1f}  |  Median: {median_count:.1f}  |  Min: {min_count}  |  Max: {max_count}'
    ax1.text(0.5, 0.98, stats_text, transform=ax1.transAxes, 
             ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Histogram
    ax2.hist(counts, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Overlapping Images', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Distribution of Overlap Counts', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    ax2.axvline(median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved overlap counts visualization to: {output_file}")
    print(f"  Total images: {len(sorted_images)}")
    print(f"  Images with overlaps: {sum(1 for c in counts if c > 0)}")
    print(f"  Mean overlaps per image: {mean_count:.1f}")
    print(f"  Median overlaps per image: {median_count:.1f}")
    print(f"  Min overlaps: {min_count}")
    print(f"  Max overlaps: {max_count}")
    
    # Also print some examples
    print(f"\nExamples:")
    print(f"  Image 0001: {overlap_counts.get(sorted_images[0] if sorted_images else '', 0)} overlaps")
    print(f"  Image 0050: {overlap_counts.get(sorted_images[49] if len(sorted_images) > 49 else '', 0)} overlaps")
    print(f"  Image 0100: {overlap_counts.get(sorted_images[99] if len(sorted_images) > 99 else '', 0)} overlaps")
    print(f"  Image 0155: {overlap_counts.get(sorted_images[-1] if sorted_images else '', 0)} overlaps")


if __name__ == "__main__":
    visualize_overlap_counts()
