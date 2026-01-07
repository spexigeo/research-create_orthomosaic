"""
Test script to demonstrate scanline-based matching improvements.
"""
import json
from pathlib import Path
from scanline_matching import ScanlineMatcher, load_scanlines_from_analysis, parse_image_number
import re


def parse_image_number(image_name: str) -> int:
    """Extract image number from filename."""
    match = re.search(r'_(\d{4})\.jpg$', image_name)
    return int(match.group(1)) if match else -1


def analyze_current_matches():
    """Analyze current matches to see how many are within scanline constraints."""
    # Load scanlines
    scanlines = load_scanlines_from_analysis()
    matcher = ScanlineMatcher(scanlines)
    
    # Load current matches
    matches_file = "outputs/matches_cache_cell_8928d89ac57ffff.json"
    with open(matches_file, 'r') as f:
        matches = json.load(f)
    
    print(f"Analyzing {len(matches)} current matches...")
    
    # Analyze match priorities
    priority_counts = {}
    valid_matches = 0
    invalid_matches = 0
    
    for match in matches:
        img0_path = match['image0']
        img1_path = match['image1']
        
        img0_num = parse_image_number(Path(img0_path).name)
        img1_num = parse_image_number(Path(img1_path).name)
        
        priority = matcher.get_matching_priority(img0_num, img1_num)
        should_match = matcher.should_match(img0_num, img1_num)
        
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if should_match:
            valid_matches += 1
        else:
            invalid_matches += 1
    
    print(f"\nMatch Priority Distribution:")
    for priority in sorted(priority_counts.keys()):
        count = priority_counts[priority]
        pct = 100 * count / len(matches)
        priority_name = {
            0: "Same scanline, adjacent",
            1: "Same scanline, within 2",
            2: "Same scanline, within 5",
            3: "Adjacent scanlines, similar pos",
            4: "Adjacent scanlines, any pos",
            5: "Same scanline, far",
            10: "Distant scanlines (should skip)"
        }.get(priority, f"Priority {priority}")
        print(f"  {priority_name}: {count} pairs ({pct:.1f}%)")
    
    print(f"\nSummary:")
    print(f"  Valid matches (should match): {valid_matches} ({100*valid_matches/len(matches):.1f}%)")
    print(f"  Invalid matches (should skip): {invalid_matches} ({100*invalid_matches/len(matches):.1f}%)")
    
    # Show examples of invalid matches
    print(f"\nExamples of matches that should be skipped (distant scanlines):")
    count = 0
    for match in matches:
        img0_num = parse_image_number(Path(match['image0']).name)
        img1_num = parse_image_number(Path(match['image1']).name)
        
        if not matcher.should_match(img0_num, img1_num):
            scanline0 = matcher.get_scanline(img0_num)
            scanline1 = matcher.get_scanline(img1_num)
            print(f"  Image {img0_num:04d} (scanline {scanline0}) <-> Image {img1_num:04d} (scanline {scanline1})")
            count += 1
            if count >= 10:
                break


def show_improved_matching_strategy():
    """Show how scanline-based matching would improve the matching process."""
    scanlines = load_scanlines_from_analysis()
    matcher = ScanlineMatcher(scanlines)
    
    # Get all image numbers from the central cell
    image_dir = Path("/Users/mauriciohessflores/Documents/Code/MyCode/research-qualicum_beach_gcp_analysis/input/images")
    central_cell = "8928d89ac57ffff"
    image_files = sorted([f for f in image_dir.glob(f"{central_cell}_*.jpg")])
    image_numbers = [parse_image_number(f.name) for f in image_files]
    
    print(f"\nImproved Matching Strategy for {len(image_numbers)} images:")
    print(f"  Total possible pairs: {len(image_numbers) * (len(image_numbers) - 1) // 2}")
    
    # Get priority pairs
    priority_pairs = matcher.get_priority_pairs(image_numbers)
    
    print(f"\nPriority-based pair selection:")
    priority_limits = {
        0: 200,  # Same scanline, adjacent
        1: 300,  # Same scanline, within 2
        2: 400,  # Same scanline, within 5
        3: 200,  # Adjacent scanlines, similar pos
        4: 300,  # Adjacent scanlines, any pos
    }
    
    selected_pairs = []
    for img1, img2, priority in priority_pairs:
        if priority < 5:  # Only high-priority pairs
            limit = priority_limits.get(priority, 100)
            if len([p for p in selected_pairs if p[2] == priority]) < limit:
                selected_pairs.append((img1, img2, priority))
    
    print(f"  Selected {len(selected_pairs)} high-priority pairs (vs 1000 random pairs)")
    
    # Count by priority
    priority_counts = {}
    for _, _, priority in selected_pairs:
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print(f"\n  Priority distribution:")
    for priority in sorted(priority_counts.keys()):
        count = priority_counts[priority]
        priority_name = {
            0: "Same scanline, adjacent",
            1: "Same scanline, within 2",
            2: "Same scanline, within 5",
            3: "Adjacent scanlines, similar pos",
            4: "Adjacent scanlines, any pos",
        }.get(priority, f"Priority {priority}")
        print(f"    {priority_name}: {count} pairs")
    
    print(f"\n  Estimated improvement:")
    print(f"    - Better match quality (matching images that should overlap)")
    print(f"    - Faster processing (fewer pairs, but better ones)")
    print(f"    - More accurate tracks (features matched between correct images)")


if __name__ == "__main__":
    print("="*60)
    print("Scanline-Based Matching Analysis")
    print("="*60)
    
    analyze_current_matches()
    show_improved_matching_strategy()
