"""
Visualize tracks on triplets of images from the tracks file.
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


def visualize_tracks_from_file(image_dir: str, features_data: dict, tracks_file: str, 
                                img_nums: tuple, output_path: str, only_three_image_tracks: bool = False):
    """
    Visualize tracks from tracks_cell_*.json file for a triplet of images.
    Each track is drawn in a different color.
    
    Args:
        image_dir: Directory containing quarter-resolution images
        features_data: Dictionary mapping image names to feature data
        tracks_file: Path to tracks JSON file
        img_nums: Tuple of three image numbers (e.g., (1, 2, 3) for 0001, 0002, 0003)
        output_path: Path to save PNG
        only_three_image_tracks: If True, only show tracks that span all three images
    """
    img1_num, img2_num, img3_num = img_nums
    
    # Load tracks
    with open(tracks_file, 'r') as f:
        tracks_data = json.load(f)
    
    tracks = tracks_data.get('tracks', [])
    print(f"Loaded {len(tracks)} tracks from {tracks_file}")
    
    # Find quarter-resolution images
    image_dir_path = Path(image_dir)
    img1_path = None
    img2_path = None
    img3_path = None
    
    img1_pattern = f"_{img1_num:04d}.jpg"
    img2_pattern = f"_{img2_num:04d}.jpg"
    img3_pattern = f"_{img3_num:04d}.jpg"
    
    for img_file in image_dir_path.glob('quarter_*.jpg'):
        if img1_pattern in img_file.name:
            img1_path = img_file
        elif img2_pattern in img_file.name:
            img2_path = img_file
        elif img3_pattern in img_file.name:
            img3_path = img_file
    
    if not all([img1_path, img2_path, img3_path]):
        print(f"Warning: Could not find all three images for {img_nums}")
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
    
    # Get feature keys
    img1_feat_name = None
    img2_feat_name = None
    img3_feat_name = None
    
    for feat_name in features_data.keys():
        if img1_pattern in feat_name or feat_name.endswith(img1_pattern):
            img1_feat_name = feat_name
        elif img2_pattern in feat_name or feat_name.endswith(img2_pattern):
            img2_feat_name = feat_name
        elif img3_pattern in feat_name or feat_name.endswith(img3_pattern):
            img3_feat_name = feat_name
    
    if not all([img1_feat_name, img2_feat_name, img3_feat_name]):
        print(f"Warning: Could not find feature keys for all images")
        return
    
    # Get features
    feat1 = features_data.get(img1_feat_name, {})
    feat2 = features_data.get(img2_feat_name, {})
    feat3 = features_data.get(img3_feat_name, {})
    
    if not all([feat1, feat2, feat3]):
        print(f"Warning: Could not find features for all images")
        return
    
    kpts1 = np.array(feat1['keypoints'])
    kpts2 = np.array(feat2['keypoints'])
    kpts3 = np.array(feat3['keypoints'])
    
    # Filter tracks to only those involving our three images
    # Track format: {'features': [('image_name', feature_idx), ...], ...}
    # Only count tracks that have at least 2 features within our 3 images, and all features are valid
    relevant_tracks = []
    img1_name_clean = img1_path.name.replace('quarter_', '')
    img2_name_clean = img2_path.name.replace('quarter_', '')
    img3_name_clean = img3_path.name.replace('quarter_', '')
    
    for track in tracks:
        track_features = track.get('features', [])
        # Map track features to our image indices (only for features in our 3 images)
        mapped_track = []
        for img_name, feat_idx in track_features:
            # Normalize image name (handle with/without quarter_ prefix)
            img_name_clean = img_name.replace('quarter_', '')
            if img_name_clean == img1_name_clean or img_name == img1_feat_name:
                # Validate feature index
                if feat_idx < len(kpts1):
                    mapped_track.append((0, feat_idx))  # Image 0
            elif img_name_clean == img2_name_clean or img_name == img2_feat_name:
                # Validate feature index
                if feat_idx < len(kpts2):
                    mapped_track.append((1, feat_idx))  # Image 1
            elif img_name_clean == img3_name_clean or img_name == img3_feat_name:
                # Validate feature index
                if feat_idx < len(kpts3):
                    mapped_track.append((2, feat_idx))  # Image 2
        
        # Only include tracks that have at least 2 features in our 3 images
        # This ensures we're counting actual tracks (2-3 features), not individual features
        if len(mapped_track) >= 2:
            relevant_tracks.append(mapped_track)
    
    # Count tracks by length (2-image vs 3-image tracks) before filtering
    two_image_tracks = sum(1 for t in relevant_tracks if len(t) == 2)
    three_image_tracks = sum(1 for t in relevant_tracks if len(t) == 3)
    
    # Filter to only 3-image tracks if requested
    if only_three_image_tracks:
        relevant_tracks = [t for t in relevant_tracks if len(t) == 3]
        print(f"Found {len(relevant_tracks)} three-image tracks in images {img1_num:04d}, {img2_num:04d}, {img3_num:04d}")
    else:
        print(f"Found {len(relevant_tracks)} tracks with 2+ features in images {img1_num:04d}, {img2_num:04d}, {img3_num:04d}")
        print(f"  ({two_image_tracks} two-image tracks, {three_image_tracks} three-image tracks)")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    axes[0].imshow(img1_rgb)
    axes[0].axis('off')
    axes[0].set_title(f'{img1_path.name}', fontsize=14, fontweight='bold')
    
    axes[1].imshow(img2_rgb)
    axes[1].axis('off')
    axes[1].set_title(f'{img2_path.name}', fontsize=14, fontweight='bold')
    
    axes[2].imshow(img3_rgb)
    axes[2].axis('off')
    axes[2].set_title(f'{img3_path.name}', fontsize=14, fontweight='bold')
    
    # Generate colors for each track
    import matplotlib.cm as cm
    if len(relevant_tracks) > 0:
        num_colors = min(20, len(relevant_tracks))
        if len(relevant_tracks) > 20:
            num_colors = len(relevant_tracks)
        colors = plt.get_cmap('tab20', num_colors)
        
        # Draw each track in its own color
        from matplotlib.patches import ConnectionPatch
        for track_idx, track in enumerate(relevant_tracks):
            color = colors(track_idx % colors.N)
            
            # Draw dots for each feature in the track
            for img_idx, feat_idx in track:
                if img_idx == 0:
                    if feat_idx < len(kpts1):
                        pt = kpts1[feat_idx]
                        axes[0].plot(pt[0], pt[1], 'o', color=color, markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.9, zorder=10)
                elif img_idx == 1:
                    if feat_idx < len(kpts2):
                        pt = kpts2[feat_idx]
                        axes[1].plot(pt[0], pt[1], 'o', color=color, markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.9, zorder=10)
                elif img_idx == 2:
                    if feat_idx < len(kpts3):
                        pt = kpts3[feat_idx]
                        axes[2].plot(pt[0], pt[1], 'o', color=color, markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1.5, alpha=0.9, zorder=10)
            
            # Draw lines connecting features in the track
            for i in range(len(track) - 1):
                img_idx1, feat_idx1 = track[i]
                img_idx2, feat_idx2 = track[i + 1]
                
                # Get points and axes
                if img_idx1 == 0:
                    if feat_idx1 >= len(kpts1):
                        continue
                    pt1 = kpts1[feat_idx1]
                    ax1 = axes[0]
                elif img_idx1 == 1:
                    if feat_idx1 >= len(kpts2):
                        continue
                    pt1 = kpts2[feat_idx1]
                    ax1 = axes[1]
                else:
                    if feat_idx1 >= len(kpts3):
                        continue
                    pt1 = kpts3[feat_idx1]
                    ax1 = axes[2]
                
                if img_idx2 == 0:
                    if feat_idx2 >= len(kpts1):
                        continue
                    pt2 = kpts1[feat_idx2]
                    ax2 = axes[0]
                elif img_idx2 == 1:
                    if feat_idx2 >= len(kpts2):
                        continue
                    pt2 = kpts2[feat_idx2]
                    ax2 = axes[1]
                else:
                    if feat_idx2 >= len(kpts3):
                        continue
                    pt2 = kpts3[feat_idx2]
                    ax2 = axes[2]
                
                # Draw line connecting the two features
                line = ConnectionPatch((pt1[0], pt1[1]), (pt2[0], pt2[1]), 
                                      "data", "data",
                                      axesA=ax1, axesB=ax2,
                                      arrowstyle="-", color=color, linewidth=2.5, alpha=0.8, zorder=5)
                fig.add_artist(line)
    
    if only_three_image_tracks:
        title = f'Three-image tracks only (images {img1_num:04d}, {img2_num:04d}, {img3_num:04d}): {len(relevant_tracks)} tracks'
    else:
        title = f'Tracks from tracks file (images {img1_num:04d}, {img2_num:04d}, {img3_num:04d}): {len(relevant_tracks)} tracks'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved track visualization to: {output_path}")


def main():
    # Paths
    quarter_res_dir = "inputs/quarter_resolution_images"
    image_dir = quarter_res_dir
    
    # Try to find features file
    features_file = None
    for fname in ["outputs/features_cell_8928d89ac57ffff.json", "outputs/features_cache_cell_8928d89ac57ffff.json"]:
        if Path(fname).exists():
            features_file = fname
            break
    
    if not features_file:
        print("Error: Could not find features file")
        return
    
    # Try to find tracks file
    tracks_file = Path("outputs/tracks_cell_8928d89ac57ffff.json")
    if not tracks_file.exists():
        print("Error: Could not find tracks file")
        return
    
    output_dir = Path("outputs/visualize_tracks_on_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using features file: {features_file}")
    print(f"Using tracks file: {tracks_file}")
    
    # Load data
    print("Loading features...")
    features_data = load_features(features_file)
    print(f"Loaded features for {len(features_data)} images")
    
    # Process multiple triplets
    triplets = [
        (1, 2, 3),    # 0001, 0002, 0003
        (51, 52, 53), # 0051, 0052, 0053
        (101, 102, 103), # 0101, 0102, 0103
        (151, 152, 153)  # 0151, 0152, 0153
    ]
    
    for img1_num, img2_num, img3_num in triplets:
        print(f"\nProcessing triplet: {img1_num:04d}, {img2_num:04d}, {img3_num:04d}")
        # All tracks (2-image and 3-image)
        output_path = output_dir / f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}.png"
        visualize_tracks_from_file(
            image_dir, features_data, str(tracks_file),
            (img1_num, img2_num, img3_num), str(output_path), only_three_image_tracks=False
        )
        # Only 3-image tracks
        output_path_three = output_dir / f"tracks_{img1_num:04d}_{img2_num:04d}_{img3_num:04d}_three_image_only.png"
        visualize_tracks_from_file(
            image_dir, features_data, str(tracks_file),
            (img1_num, img2_num, img3_num), str(output_path_three), only_three_image_tracks=True
        )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
