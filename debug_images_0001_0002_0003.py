"""
Debug script to visualize features and matches for images 0001, 0002, and 0003.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2


def load_features(features_file: str):
    """Load features from JSON file."""
    with open(features_file, 'r') as f:
        return json.load(f)


def load_matches(matches_file: str):
    """Load matches from JSON file."""
    with open(matches_file, 'r') as f:
        return json.load(f)


def load_footprint_overlaps():
    """Load footprint overlaps."""
    overlaps_file = Path('outputs/footprint_overlaps.json')
    if overlaps_file.exists():
        with open(overlaps_file, 'r') as f:
            data = json.load(f)
            # Return the overlaps list if it exists, otherwise return empty dict
            if 'overlaps' in data and isinstance(data['overlaps'], list):
                # Convert list to dict for easier lookup
                overlaps_dict = {}
                for entry in data['overlaps']:
                    if isinstance(entry, dict):
                        # Handle 'image1'/'image2' format (from compute_footprint_overlap.py)
                        img0 = entry.get('image1')  # First image
                        img1 = entry.get('image2')  # Second image
                        if img0 and img1:
                            key = (img0, img1)
                            overlaps_dict[key] = entry.get('overlap_percentage', 0.0)
                            # Also add reverse key for easier lookup
                            overlaps_dict[(img1, img0)] = entry.get('overlap_percentage', 0.0)
                return overlaps_dict
            return data
    return {}


def visualize_features_on_image(image_path: str, features_data: dict, output_path: str, image_name: str):
    """
    Visualize features on a single image.
    
    Args:
        image_path: Path to quarter-resolution image
        features_data: Feature data dict with 'keypoints' and 'scores'
        output_path: Path to save PNG
        image_name: Name of image (for title)
    """
    # Load image (quarter-resolution)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get features (already in quarter-resolution coordinates)
    keypoints = np.array(features_data['keypoints'])
    scores = np.array(features_data['scores'])
    
    # Use coordinates as-is (already in quarter-resolution space)
    keypoints_plot = keypoints
    
    # Sort by score (descending) to get ranking
    sorted_indices = np.argsort(scores)[::-1]  # Highest score first
    rank_map = {idx: rank+1 for rank, idx in enumerate(sorted_indices)}  # 1 = highest score
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Plot features as red dots with labels
    for i, (kpt, score) in enumerate(zip(keypoints_plot, scores)):
        rank = rank_map[i]
        x, y = kpt[0], kpt[1]
        
        # Draw red dot
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1)
        
        # Add label (rank 1-100)
        ax.text(x + 10, y - 10, str(rank), fontsize=8, color='yellow', 
                weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    ax.set_title(f'{image_name}\n100 Features (labeled by score rank: 1=highest, 100=lowest)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature visualization to: {output_path}")


def visualize_matches(image_dir: str, features_data: dict, matches_data: list, output_path: str):
    """
    Visualize matches between images 0001->0002 (green) and 0002->0003 (blue).
    
    Args:
        image_dir: Directory containing quarter-resolution images
        features_data: Dictionary mapping image names to feature data
        matches_data: List of match dictionaries
        output_path: Path to save PNG
    """
    image_dir_path = Path(image_dir)
    
    # Find quarter-resolution images (they have "quarter_" prefix)
    img1_path = None
    img2_path = None
    img3_path = None
    
    for img_file in image_dir_path.glob('quarter_*.jpg'):
        if '_0001.jpg' in img_file.name:
            img1_path = img_file
        elif '_0002.jpg' in img_file.name:
            img2_path = img_file
        elif '_0003.jpg' in img_file.name:
            img3_path = img_file
    
    if not all([img1_path, img2_path, img3_path]):
        print(f"Warning: Could not find all three images")
        print(f"  Found: {[p.name if p else None for p in [img1_path, img2_path, img3_path]]}")
        return
    
    # Load images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    img3 = cv2.imread(str(img3_path))
    
    if any(img is None for img in [img1, img2, img3]):
        print("Warning: Could not load one or more images")
        return
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    
    # Get image names for feature lookup (need to find the correct keys)
    # Find feature keys that match these images
    img1_feat_name = None
    img2_feat_name = None
    img3_feat_name = None
    
    for feat_name in features_data.keys():
        if img1_path.name in feat_name or feat_name.endswith('_0001.jpg'):
            img1_feat_name = feat_name
        elif img2_path.name in feat_name or feat_name.endswith('_0002.jpg'):
            img2_feat_name = feat_name
        elif img3_path.name in feat_name or feat_name.endswith('_0003.jpg'):
            img3_feat_name = feat_name
    
    if not all([img1_feat_name, img2_feat_name, img3_feat_name]):
        print(f"Warning: Could not find feature keys for all images")
        print(f"  Found: {[img1_feat_name, img2_feat_name, img3_feat_name]}")
        return
    
    # Get features
    feat1 = features_data.get(img1_feat_name, {})
    feat2 = features_data.get(img2_feat_name, {})
    feat3 = features_data.get(img3_feat_name, {})
    
    if not all([feat1, feat2, feat3]):
        print(f"Warning: Could not find features for all images")
        print(f"  Found: {[img1_name in features_data, img2_name in features_data, img3_name in features_data]}")
        return
    
    # Use features as-is (already in quarter-resolution coordinates)
    kpts1 = np.array(feat1['keypoints'])
    kpts2 = np.array(feat2['keypoints'])
    kpts3 = np.array(feat3['keypoints'])
    
    # Find matches 0001->0002
    matches_12 = []
    matches_23 = []
    
    for match in matches_data:
        img0 = match.get('image0', '')
        img1 = match.get('image1', '')
        
        # Check for matches 0001->0002 (exact match on feature names)
        if img0 == img1_feat_name and img1 == img2_feat_name:
            matches_12 = np.array(match['matches'])
            print(f"Found 0001->0002: {len(matches_12)} matches")
        # Check for matches 0002->0003 (exact match on feature names)
        elif img0 == img2_feat_name and img1 == img3_feat_name:
            matches_23 = np.array(match['matches'])
            print(f"Found 0002->0003: {len(matches_23)} matches")
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # Get image dimensions for proper spacing
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    h3, w3 = img3_rgb.shape[:2]
    
    # Image 1
    axes[0].imshow(img1_rgb)
    axes[0].axis('off')
    axes[0].set_title(f'{img1_path.name}', fontsize=14, fontweight='bold')
    
    # Image 2
    axes[1].imshow(img2_rgb)
    axes[1].axis('off')
    axes[1].set_title(f'{img2_path.name}', fontsize=14, fontweight='bold')
    
    # Image 3
    axes[2].imshow(img3_rgb)
    axes[2].axis('off')
    axes[2].set_title(f'{img3_path.name}', fontsize=14, fontweight='bold')
    
    # Filter matches to only valid indices (matches may have been computed with different feature set)
    valid_matches_12 = []
    valid_matches_23 = []
    
    if len(matches_12) > 0:
        for match in matches_12:
            idx1, idx2 = int(match[0]), int(match[1])
            if idx1 < len(kpts1) and idx2 < len(kpts2):
                valid_matches_12.append(match)
    
    if len(matches_23) > 0:
        for match in matches_23:
            idx2, idx3 = int(match[0]), int(match[1])
            if idx2 < len(kpts2) and idx3 < len(kpts3):
                valid_matches_23.append(match)
    
    print(f"Valid matches: 0001->0002: {len(valid_matches_12)}/{len(matches_12)}, 0002->0003: {len(valid_matches_23)}/{len(matches_23)}")
    
    # Draw matches 0001->0002 (green dots)
    if len(valid_matches_12) > 0:
        print(f"Drawing {len(valid_matches_12)} valid matches between 0001 and 0002")
        for match in valid_matches_12:
            idx1, idx2 = int(match[0]), int(match[1])
            pt1 = kpts1[idx1]
            pt2 = kpts2[idx2]
            # Draw green dots on each image (larger, more visible)
            axes[0].plot(pt1[0], pt1[1], 'go', markersize=10, markeredgecolor='darkgreen', 
                       markeredgewidth=2, alpha=0.9, zorder=10)
            axes[1].plot(pt2[0], pt2[1], 'go', markersize=10, markeredgecolor='darkgreen', 
                       markeredgewidth=2, alpha=0.9, zorder=10)
    else:
        print(f"WARNING: No valid matches found for 0001->0002 (all {len(matches_12)} matches have invalid indices)")
        print(f"  This suggests matches were computed with a different feature set")
        print(f"  Keypoint array lengths: {len(kpts1)}, {len(kpts2)}")
        if len(matches_12) > 0:
            sample = matches_12[0]
            print(f"  Sample match indices: {sample[0]} -> {sample[1]}")
    
    # Draw matches 0002->0003 (blue dots)
    if len(valid_matches_23) > 0:
        print(f"Drawing {len(valid_matches_23)} valid matches between 0002 and 0003")
        for match in valid_matches_23:
            idx2, idx3 = int(match[0]), int(match[1])
            pt2 = kpts2[idx2]
            pt3 = kpts3[idx3]
            # Draw blue dots on each image (larger, more visible)
            axes[1].plot(pt2[0], pt2[1], 'bo', markersize=10, markeredgecolor='darkblue', 
                       markeredgewidth=2, alpha=0.9, zorder=10)
            axes[2].plot(pt3[0], pt3[1], 'bo', markersize=10, markeredgecolor='darkblue', 
                       markeredgewidth=2, alpha=0.9, zorder=10)
    else:
        print(f"WARNING: No valid matches found for 0002->0003 (all {len(matches_23)} matches have invalid indices)")
        print(f"  This suggests matches were computed with a different feature set")
        print(f"  Keypoint array lengths: {len(kpts2)}, {len(kpts3)}")
        if len(matches_23) > 0:
            sample = matches_23[0]
            print(f"  Sample match indices: {sample[0]} -> {sample[1]}")
    
    plt.suptitle('Matches: 0001->0002 (green), 0002->0003 (blue)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved match visualization to: {output_path}")
    print(f"  Found {len(matches_12)} matches between 0001 and 0002")
    print(f"  Found {len(matches_23)} matches between 0002 and 0003")


def save_overlap_info(overlaps: dict, output_path: str, reference_image: str = "0001", features_data: dict = None):
    """
    Save overlap percentages for images that overlap with reference_image.
    
    Args:
        overlaps: Dictionary of overlaps (can be tuple keys or string keys)
        output_path: Path to save output file
        reference_image: Reference image number (e.g., "0001")
    """
    # Find all overlaps involving the reference image
    relevant_overlaps = []
    
    # Find the actual image name that contains the reference number
    # The overlaps use image names like '8928d89ac57ffff_172550_0001.jpg'
    reference_image_name = None
    if features_data:
        for feat_name in features_data.keys():
            if reference_image in feat_name or feat_name.endswith(f'_{reference_image}.jpg'):
                # Extract the base name (remove quarter_ prefix)
                reference_image_name = feat_name.replace('quarter_', '')
                print(f"Found reference image name: {reference_image_name}")
                break
    
    for key, overlap_pct in overlaps.items():
        # Handle different key formats
        if isinstance(key, tuple):
            img0, img1 = key
        elif isinstance(key, str):
            # Try to parse string key (might be "img0,img1" or similar)
            if ',' in key:
                img0, img1 = key.split(',', 1)
            else:
                continue
        else:
            continue
        
        # Check if reference image name matches either image
        img0_str = str(img0)
        img1_str = str(img1)
        
        # Skip self-overlaps
        if img0_str == img1_str:
            continue
        
        # Check if reference image name matches either image (exact match)
        matches_ref = False
        if reference_image_name:
            # Exact match or substring match
            if img0_str == reference_image_name or reference_image_name in img0_str:
                matches_ref = True
                other_img = img1_str
            elif img1_str == reference_image_name or reference_image_name in img1_str:
                matches_ref = True
                other_img = img0_str
        else:
            # Fallback: check if reference number is in image name
            if reference_image in img0_str and reference_image not in img1_str:
                matches_ref = True
                other_img = img1_str
            elif reference_image in img1_str and reference_image not in img0_str:
                matches_ref = True
                other_img = img0_str
        
        if matches_ref:
            
            # Convert overlap_pct to float if needed
            if isinstance(overlap_pct, (int, float)):
                overlap_val = float(overlap_pct)
            else:
                continue
            
            if overlap_val > 0:
                relevant_overlaps.append({
                    'image': other_img,
                    'overlap_percentage': round(overlap_val, 2)
                })
    
    # Remove duplicates (keep first occurrence)
    seen = set()
    unique_overlaps = []
    for overlap in relevant_overlaps:
        if overlap['image'] not in seen:
            seen.add(overlap['image'])
            unique_overlaps.append(overlap)
    relevant_overlaps = unique_overlaps
    
    # Sort by overlap percentage (descending)
    relevant_overlaps.sort(key=lambda x: x['overlap_percentage'], reverse=True)
    
    # Save to JSON
    output_data = {
        'reference_image': reference_image,
        'total_overlapping_images': len(relevant_overlaps),
        'overlaps': relevant_overlaps
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved overlap information to: {output_path}")
    print(f"  Found {len(relevant_overlaps)} images overlapping with {reference_image} (excluding 0%)")
    
    # Also print summary
    if relevant_overlaps:
        print(f"  Top 5 overlaps:")
        for overlap in relevant_overlaps[:5]:
            print(f"    {overlap['image']}: {overlap['overlap_percentage']}%")


def main():
    # Paths
    # Use quarter-resolution images directory
    quarter_res_dir = "inputs/quarter_resolution_images"
    image_dir = quarter_res_dir  # Use quarter-resolution images
    
    # Try to find features file (could be features_cell or features_cache)
    features_file = None
    for fname in ["outputs/features_cell_8928d89ac57ffff.json", "outputs/features_cache_cell_8928d89ac57ffff.json"]:
        if Path(fname).exists():
            features_file = fname
            break
    
    if not features_file:
        print("Error: Could not find features file")
        print("  Tried: outputs/features_cell_8928d89ac57ffff.json")
        print("  Tried: outputs/features_cache_cell_8928d89ac57ffff.json")
        return
    
    # Try to find matches file
    matches_file = None
    for fname in ["outputs/matches_unfiltered_cell_8928d89ac57ffff.json", "outputs/matches_cache_cell_8928d89ac57ffff.json"]:
        if Path(fname).exists():
            matches_file = fname
            break
    
    if not matches_file:
        print("Warning: Could not find matches file, will skip match visualization")
        matches_file = None
    
    output_dir = Path("test_image_0001")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Using features file: {features_file}")
    print(f"Using matches file: {matches_file}")
    
    # Load data
    print("Loading features...")
    features_data = load_features(features_file)
    print(f"Loaded features for {len(features_data)} images")
    
    matches_data = []
    if matches_file:
        print("Loading matches...")
        matches_data = load_matches(matches_file)
        print(f"Loaded {len(matches_data)} match pairs")
    else:
        print("Skipping matches (file not found)")
    
    print("Loading footprint overlaps...")
    overlaps = load_footprint_overlaps()
    print(f"Loaded {len(overlaps)} overlap entries")
    
    # Find images 0001, 0002, 0003
    # First, check what image names are in the features data
    feature_image_names = list(features_data.keys())
    print(f"\nSample feature keys (first 5): {feature_image_names[:5]}")
    
    # Find images that match the pattern (look for ones ending in _0001.jpg, _0002.jpg, _0003.jpg)
    img1_name = None
    img2_name = None
    img3_name = None
    
    for feat_name in feature_image_names:
        if feat_name.endswith('_0001.jpg') or '_0001.jpg' in feat_name:
            img1_name = feat_name
        elif feat_name.endswith('_0002.jpg') or '_0002.jpg' in feat_name:
            img2_name = feat_name
        elif feat_name.endswith('_0003.jpg') or '_0003.jpg' in feat_name:
            img3_name = feat_name
    
    if not all([img1_name, img2_name, img3_name]):
        print(f"Error: Could not find all three images in features")
        print(f"  Found: {[img1_name, img2_name, img3_name]}")
        return
    
    print(f"\nFound feature keys:")
    print(f"  0001: {img1_name}")
    print(f"  0002: {img2_name}")
    print(f"  0003: {img3_name}")
    
    # Find quarter-resolution image files (they have "quarter_" prefix)
    quarter_res_dir = Path("inputs/quarter_resolution_images")
    img1_path = quarter_res_dir / img1_name
    img2_path = quarter_res_dir / img2_name
    img3_path = quarter_res_dir / img3_name
    
    if not all([img1_path.exists(), img2_path.exists(), img3_path.exists()]):
        print(f"Warning: Some quarter-resolution image files not found")
        print(f"  {img1_name}: {img1_path.exists()}")
        print(f"  {img2_name}: {img2_path.exists()}")
        print(f"  {img3_name}: {img3_path.exists()}")
        return
    
    feat1 = features_data.get(img1_name)
    feat2 = features_data.get(img2_name)
    feat3 = features_data.get(img3_name)
    
    if not all([feat1, feat2, feat3]):
        print(f"Error: Could not find features for all images")
        print(f"  Found: {[img1_name in features_data, img2_name in features_data, img3_name in features_data]}")
        print(f"  Available keys (first 5): {list(features_data.keys())[:5]}")
        return
    
    # 1. Visualize features on each image
    print("\n1. Creating feature visualizations...")
    visualize_features_on_image(
        str(img1_path), feat1, 
        str(output_dir / "image_0001_features.png"), 
        img1_path.name
    )
    visualize_features_on_image(
        str(img2_path), feat2, 
        str(output_dir / "image_0002_features.png"), 
        img2_path.name
    )
    visualize_features_on_image(
        str(img3_path), feat3, 
        str(output_dir / "image_0003_features.png"), 
        img3_path.name
    )
    
    # 2. Visualize matches
    if matches_data:
        print("\n2. Creating match visualization...")
        visualize_matches(
            image_dir, features_data, matches_data,
            str(output_dir / "matches_0001_0002_0003.png")
        )
    else:
        print("\n2. Skipping match visualization (no matches file)")
    
    # 3. Save overlap information
    print("\n3. Saving overlap information...")
    save_overlap_info(
        overlaps, 
        str(output_dir / "overlaps_with_0001.json"),
        reference_image="0001",
        features_data=features_data
    )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
