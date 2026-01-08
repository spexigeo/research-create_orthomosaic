"""
Visualize a specific track by showing feature locations on images.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from collections import defaultdict


def visualize_track(track_id: int, num_images: int = 5, 
                    tracks_file: str = "outputs/tracks_cell_8928d89ac57ffff.json",
                    features_file: str = "outputs/features_cache_cell_8928d89ac57ffff.json",
                    matches_file: str = "outputs/matches_cache_cell_8928d89ac57ffff.json",
                    image_dir: str = "inputs/quarter_resolution_images",
                    output_dir: str = "outputs/visualize_single_track",
                    verify_matches: bool = True):
    """
    Visualize a track by showing feature locations on the first N images.
    
    Args:
        track_id: ID of the track to visualize (or -1 for longest track)
        num_images: Number of images to show
        tracks_file: Path to tracks JSON file
        features_file: Path to features cache JSON file
        image_dir: Directory containing original images
        output_dir: Directory to save visualization
    """
    # Load tracks
    with open(tracks_file, 'r') as f:
        tracks_data = json.load(f)
    
    # Find the track
    if track_id == -1:
        # Find longest track
        track = max(tracks_data['tracks'], key=lambda t: t['length'])
        track_id = track['track_id']
        print(f"Found longest track: ID {track_id}, length {track['length']}")
    else:
        track = next(t for t in tracks_data['tracks'] if t['track_id'] == track_id)
        print(f"Visualizing track: ID {track_id}, length {track['length']}")
    
    # Load features (try features_cell first, then fall back to features_cache)
    features_data = None
    features_cell_file = features_file.replace('features_cache', 'features').replace('_cache', '')
    if Path(features_cell_file).exists():
        try:
            with open(features_cell_file, 'r') as f:
                features_data = json.load(f)
            print(f"Loaded features from: {features_cell_file}")
        except:
            pass
    
    if features_data is None:
        if Path(features_file).exists():
            with open(features_file, 'r') as f:
                features_data = json.load(f)
            print(f"Loaded features from: {features_file}")
        else:
            raise FileNotFoundError(f"Features file not found: {features_file} or {features_cell_file}")
    
    # Load matches for verification (optional - use unfiltered matches if available)
    matches_data = None
    if not Path(matches_file).exists():
        # Try unfiltered matches file
        unfiltered_matches = matches_file.replace('matches_cache', 'matches_unfiltered')
        if Path(unfiltered_matches).exists():
            matches_file = unfiltered_matches
            print(f"Using unfiltered matches file: {matches_file}")
    if verify_matches:
        with open(matches_file, 'r') as f:
            matches_data = json.load(f)
        
        # Build match lookup: (img0, img1) -> dict of {feat0_idx: feat1_idx}
        match_lookup = defaultdict(dict)
        for match in matches_data:
            img0 = Path(match['image0']).name
            img1 = Path(match['image1']).name
            matches_list = match['matches']
            
            # Convert to numpy array if it's a list
            if isinstance(matches_list, list):
                matches_list = np.array(matches_list)
            
            # Build forward and reverse mappings
            if len(matches_list) > 0:
                match_lookup[(img0, img1)] = {int(m[0]): int(m[1]) for m in matches_list}
                match_lookup[(img1, img0)] = {int(m[1]): int(m[0]) for m in matches_list}
    
    # Get first N images from track
    if 'features' in track:
        # Full feature information available
        track_features = track['features']  # List of (image_name, feature_idx) tuples
        images_to_show = track_features[:num_images]
    else:
        # Only image names available - need to reconstruct feature indices
        # For now, we'll use the first feature from each image (this is a limitation)
        print("Warning: Full feature information not available, using first feature from each image")
        images_to_show = [(img, 0) for img in track['images'][:num_images]]
    
    print(f"Showing first {len(images_to_show)} images from track")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load images and get feature coordinates
    fig, axes = plt.subplots(1, len(images_to_show), figsize=(5*len(images_to_show), 5))
    if len(images_to_show) == 1:
        axes = [axes]
    
    image_dir_path = Path(image_dir)
    
    for idx, feature_info in enumerate(images_to_show):
        # Handle both tuple and list formats
        if isinstance(feature_info, (list, tuple)) and len(feature_info) == 2:
            img_name, feat_idx = feature_info[0], feature_info[1]
        else:
            print(f"Warning: Unexpected feature format: {feature_info}")
            continue
        
        # Get image name - keep "quarter_" prefix if present (images are stored with this prefix)
        # Try with quarter_ prefix first, then without
        img_path = image_dir_path / img_name
        original_img_name = img_name.replace("quarter_", "")  # For display purposes
        if not img_path.exists():
            # Try without quarter_ prefix
            img_path = image_dir_path / original_img_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get feature coordinates from cache
        # Note: features cache uses quarter-resolution coordinates
        # We need to scale them up by 4x to get original coordinates
        # The cache might use full paths, so try both the image name and full path
        feat_data = None
        cache_key = None
        
        # Try exact match first
        if img_name in features_data:
            feat_data = features_data[img_name]
            cache_key = img_name
        else:
            # Try matching by filename
            img_basename = Path(img_name).name
            for key in features_data.keys():
                if Path(key).name == img_basename:
                    feat_data = features_data[key]
                    cache_key = key
                    break
        
        if feat_data is not None:
            keypoints = np.array(feat_data['keypoints'])
            
            if feat_idx < len(keypoints):
                # Get feature coordinates (already in quarter-resolution since features were extracted from quarter-res images)
                kpt = keypoints[feat_idx]
                
                print(f"Image {idx+1}: {original_img_name}")
                print(f"  Cache key used: {cache_key}")
                print(f"  Feature index: {feat_idx}")
                print(f"  Feature coords: ({kpt[0]:.1f}, {kpt[1]:.1f})")
                print(f"  Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
                
                # Verify match with previous image if available
                if verify_matches and idx > 0 and matches_data:
                    prev_feature_info = images_to_show[idx-1]
                    if isinstance(prev_feature_info, (list, tuple)) and len(prev_feature_info) == 2:
                        prev_img_name, prev_feat_idx = prev_feature_info[0], prev_feature_info[1]
                        prev_img_basename = Path(prev_img_name).name
                        curr_img_basename = Path(img_name).name
                        
                        match_dict = match_lookup.get((prev_img_basename, curr_img_basename), {})
                        if prev_feat_idx in match_dict:
                            expected_feat_idx = match_dict[prev_feat_idx]
                            if expected_feat_idx == feat_idx:
                                print(f"  ✓ Verified: matches feature {prev_feat_idx} from previous image")
                            else:
                                print(f"  ✗ WARNING: Expected feature {expected_feat_idx} (from match), but track has {feat_idx}")
                        else:
                            print(f"  ? No direct match found with previous image (might be connected through other images)")
                
                # Display image
                axes[idx].imshow(img_rgb)
                axes[idx].axis('off')
                
                # Check if this feature is directly matched to previous
                is_direct_match = False
                if verify_matches and idx > 0 and matches_data:
                    prev_feature_info = images_to_show[idx-1]
                    if isinstance(prev_feature_info, (list, tuple)) and len(prev_feature_info) == 2:
                        prev_img_name, prev_feat_idx = prev_feature_info[0], prev_feature_info[1]
                        prev_img_basename = Path(prev_img_name).name
                        curr_img_basename = Path(img_name).name
                        match_dict = match_lookup.get((prev_img_basename, curr_img_basename), {})
                        is_direct_match = (prev_feat_idx in match_dict and match_dict[prev_feat_idx] == feat_idx)
                
                # Set title with match status
                match_status = "✓ Match verified" if is_direct_match else "⚠ Match not found in matches file"
                axes[idx].set_title(f"{original_img_name}\nFeature {feat_idx} at ({kpt[0]:.0f}, {kpt[1]:.0f})\n{match_status}", 
                                   fontsize=9, color='green' if is_direct_match else 'orange')
                
                # Draw big red dot at feature location
                # Note: matplotlib imshow uses (0,0) at top-left, x increases right, y increases down
                # Features are in quarter-resolution coordinates, matching the quarter-resolution images
                color = 'red' if is_direct_match else 'orange'
                axes[idx].plot(kpt[0], kpt[1], 'o', color=color,
                              markersize=30, markeredgecolor='darkred' if is_direct_match else 'darkorange', 
                              markeredgewidth=3, alpha=0.8, label='Feature')
                # Also draw a circle for better visibility
                from matplotlib.patches import Circle
                circle = Circle((kpt[0], kpt[1]), radius=50, 
                              color=color, fill=False, linewidth=3, alpha=0.8)
                axes[idx].add_patch(circle)
            else:
                print(f"Warning: Feature index {feat_idx} out of range for {img_name}")
                axes[idx].imshow(img_rgb)
                axes[idx].axis('off')
                axes[idx].set_title(f"{original_img_name}\n(Feature not found)", fontsize=10)
        else:
            print(f"Warning: Image {img_name} not found in features cache")
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f"{original_img_name}\n(Features not found)", fontsize=10)
    
    plt.suptitle(f'Track {track_id} - Length {track["length"]} (showing first {len(images_to_show)} images)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    output_file = output_path / f"track_{track_id}_first_{len(images_to_show)}_images.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization to: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    # Visualize the longest track (length 114)
    visualize_track(track_id=-1, num_images=5)
