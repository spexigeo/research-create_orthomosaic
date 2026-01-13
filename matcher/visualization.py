"""
Visualization functions for features, matches, and tracks.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import cv2
from collections import defaultdict
import random


def visualize_features_and_matches(
    match_results: Dict,
    image_paths: List[str],
    output_path: str,
    max_pairs: int = 10,
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Visualize features and matches for image pairs.
    
    Args:
        match_results: Dictionary from LightGlueMatcher.match_cell_images()
        image_paths: List of image paths to visualize
        output_path: Path to save visualization
        max_pairs: Maximum number of pairs to visualize
        figsize: Figure size
    """
    features = match_results['features']
    matches = match_results['matches']
    
    # Filter matches to only include images in image_paths
    image_paths_set = set(image_paths)
    relevant_matches = [
        m for m in matches 
        if m['image0'] in image_paths_set and m['image1'] in image_paths_set
    ]
    
    # Sort by number of matches (descending) and take top pairs
    relevant_matches.sort(key=lambda x: x['num_matches'], reverse=True)
    relevant_matches = relevant_matches[:max_pairs]
    
    if not relevant_matches:
        print("No matches found to visualize")
        # Create an empty figure with a message
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No matches found to visualize\n(Check that image paths match between matches and image_paths)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate grid size
    n_pairs = len(relevant_matches)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, match_dict in enumerate(relevant_matches):
        ax = axes[idx]
        
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        match_indices = match_dict['matches']
        match_confidence = match_dict.get('match_confidence', None)
        
        # Check if features exist for both images
        if img0_path not in features or img1_path not in features:
            ax.text(0.5, 0.5, f'Features not found\n{Path(img0_path).name}\n{Path(img1_path).name}', 
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Load images
        try:
            img0 = np.array(Image.open(img0_path))
            img1 = np.array(Image.open(img1_path))
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading images:\n{str(e)}', 
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Get features
        feats0 = features[img0_path]
        feats1 = features[img1_path]
        
        # Create side-by-side visualization
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        
        # Resize if needed to fit
        max_h = max(h0, h1)
        max_w = max(w0, w1)
        scale = min(800 / max_h, 1200 / max_w)
        
        if scale < 1.0:
            new_h0, new_w0 = int(h0 * scale), int(w0 * scale)
            new_h1, new_w1 = int(h1 * scale), int(w1 * scale)
            img0 = cv2.resize(img0, (new_w0, new_h0))
            img1 = cv2.resize(img1, (new_w1, new_h1))
            kp_scale0 = scale
            kp_scale1 = scale
        else:
            kp_scale0 = 1.0
            kp_scale1 = 1.0
        
        # Combine images
        h_combined = max(img0.shape[0], img1.shape[0])
        w_combined = img0.shape[1] + img1.shape[1]
        combined = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
        
        combined[:img0.shape[0], :img0.shape[1]] = img0
        combined[:img1.shape[0], img0.shape[1]:] = img1
        
        ax.imshow(combined)
        ax.axis('off')
        
        # Draw keypoints
        kp0 = feats0['keypoints'] * kp_scale0
        kp1 = feats1['keypoints'] * kp_scale1
        
        # Draw all keypoints in image 0
        ax.scatter(kp0[:, 0], kp0[:, 1], c='cyan', s=10, alpha=0.6, marker='.')
        
        # Draw all keypoints in image 1 (offset)
        ax.scatter(kp1[:, 0] + img0.shape[1], kp1[:, 1], c='cyan', s=10, alpha=0.6, marker='.')
        
        # Draw matches
        for match_idx in match_indices[:100]:  # Limit to 100 matches for clarity
            idx0, idx1 = match_idx[0], match_idx[1]
            
            if idx0 < len(kp0) and idx1 < len(kp1):
                pt0 = kp0[idx0]
                pt1 = kp1[idx1]
                
                # Draw line
                ax.plot([pt0[0], pt1[0] + img0.shape[1]], 
                       [pt0[1], pt1[1]], 
                       'r-', alpha=0.3, linewidth=0.5)
                
                # Draw points
                ax.plot(pt0[0], pt0[1], 'ro', markersize=2)
                ax.plot(pt1[0] + img0.shape[1], pt1[1], 'ro', markersize=2)
        
        # Add title
        img0_name = Path(img0_path).name
        img1_name = Path(img1_path).name
        ax.set_title(f"{img0_name} <-> {img1_name}\n{len(match_indices)} matches", 
                    fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def visualize_feature_distribution(
    match_results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize feature distribution across images.
    
    Args:
        match_results: Dictionary from LightGlueMatcher.match_cell_images()
        output_path: Path to save visualization
        figsize: Figure size
    """
    features = match_results['features']
    matches = match_results['matches']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Number of features per image
    ax = axes[0, 0]
    image_names = [Path(path).name for path in features.keys()]
    num_features = [len(f['keypoints']) for f in features.values()]
    ax.bar(range(len(image_names)), num_features)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Number of Features')
    ax.set_title('Features per Image')
    ax.set_xticks(range(len(image_names)))
    ax.set_xticklabels([name[:20] + '...' if len(name) > 20 else name 
                        for name in image_names], 
                       rotation=45, ha='right', fontsize=6)
    
    # Plot 2: Number of matches per pair
    ax = axes[0, 1]
    num_matches = [m['num_matches'] for m in matches]
    ax.hist(num_matches, bins=20, edgecolor='black')
    ax.set_xlabel('Number of Matches')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Matches per Pair')
    
    # Plot 3: Match confidence distribution
    ax = axes[1, 0]
    all_confidences = []
    for m in matches:
        if 'match_confidence' in m and m['match_confidence'] is not None:
            conf_list = m['match_confidence']
            # Handle both list and numpy array
            if isinstance(conf_list, list):
                all_confidences.extend([c for c in conf_list if c is not None])
            else:
                # numpy array
                all_confidences.extend(conf_list.tolist() if hasattr(conf_list, 'tolist') else list(conf_list))
    if all_confidences and len(all_confidences) > 0:
        ax.hist(all_confidences, bins=50, edgecolor='black')
        ax.set_xlabel('Match Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Match Confidence Distribution')
    else:
        ax.text(0.5, 0.5, 'No confidence scores available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Match Confidence Distribution')
    
    # Plot 4: Total matches summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Summary Statistics:
    
    Total Images: {len(features)}
    Total Image Pairs: {len(matches)}
    Total Matches: {sum(m['num_matches'] for m in matches)}
    Average Matches per Pair: {np.mean([m['num_matches'] for m in matches]):.1f}
    Median Matches per Pair: {np.median([m['num_matches'] for m in matches]):.1f}
    Max Matches in a Pair: {max([m['num_matches'] for m in matches]) if matches else 0}
    Min Matches in a Pair: {min([m['num_matches'] for m in matches]) if matches else 0}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, 
           family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature distribution visualization to {output_path}")


def _load_features(features_file: str):
    """Load features from JSON file."""
    with open(features_file, 'r') as f:
        return json.load(f)


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
                has_indirect_match = False
                if verify_matches and idx > 0 and matches_data:
                    prev_feature_info = images_to_show[idx-1]
                    if isinstance(prev_feature_info, (list, tuple)) and len(prev_feature_info) == 2:
                        prev_img_name, prev_feat_idx = prev_feature_info[0], prev_feature_info[1]
                        prev_img_basename = Path(prev_img_name).name
                        curr_img_basename = Path(img_name).name
                        match_dict = match_lookup.get((prev_img_basename, curr_img_basename), {})
                        is_direct_match = (prev_feat_idx in match_dict and match_dict[prev_feat_idx] == feat_idx)
                        
                        # Check if there's any match between these images (even if not this specific feature pair)
                        if not is_direct_match and (prev_img_basename, curr_img_basename) in match_lookup:
                            has_indirect_match = True
                
                # Set title with match status
                if is_direct_match:
                    match_status = "✓ Direct match verified"
                    status_color = 'green'
                    dot_color = 'red'
                elif has_indirect_match:
                    match_status = "→ Connected via track (no direct match)"
                    status_color = 'blue'
                    dot_color = 'orange'
                else:
                    match_status = "→ Track connection (matches file may not have this pair)"
                    status_color = 'gray'
                    dot_color = 'orange'
                axes[idx].set_title(f"{original_img_name}\nFeature {feat_idx} at ({kpt[0]:.0f}, {kpt[1]:.0f})\n{match_status}", 
                                   fontsize=9, color=status_color)
                
                # Draw big red dot at feature location
                color = dot_color
                markeredge_color = 'darkred' if is_direct_match else ('darkorange' if has_indirect_match else 'darkgray')
                axes[idx].plot(kpt[0], kpt[1], 'o', color=color,
                              markersize=30, markeredgecolor=markeredge_color, 
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


def visualize_tracks_on_image_triplets(image_dir: str, features_data: dict, tracks_file: str, 
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
    
    # Try quarter-resolution images first, then full-resolution
    for pattern in ['quarter_*.jpg', '*.jpg']:
        for img_file in image_dir_path.glob(pattern):
            if img1_pattern in img_file.name:
                img1_path = img_file
            elif img2_pattern in img_file.name:
                img2_path = img_file
            elif img3_pattern in img_file.name:
                img3_path = img_file
        if all([img1_path, img2_path, img3_path]):
            break
    
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
    relevant_tracks = []
    img1_name_clean = img1_path.name.replace('quarter_', '')
    img2_name_clean = img2_path.name.replace('quarter_', '')
    img3_name_clean = img3_path.name.replace('quarter_', '')
    
    # Also try matching with full paths
    img1_full_name = str(img1_path)
    img2_full_name = str(img2_path)
    img3_full_name = str(img3_path)
    
    for track in tracks:
        track_features = track.get('features', [])
        # Map track features to our image indices (only for features in our 3 images)
        mapped_track = []
        for img_name, feat_idx in track_features:
            # Normalize image name (handle with/without quarter_ prefix and path variations)
            img_name_clean = img_name.replace('quarter_', '')
            img_name_base = Path(img_name).name if '/' in img_name or '\\' in img_name else img_name
            
            # Try multiple matching strategies
            matches_img1 = (img_name_clean == img1_name_clean or 
                          img_name == img1_feat_name or
                          img_name_base == img1_name_clean or
                          img_name == img1_full_name or
                          Path(img_name).name == img1_name_clean)
            
            matches_img2 = (img_name_clean == img2_name_clean or 
                          img_name == img2_feat_name or
                          img_name_base == img2_name_clean or
                          img_name == img2_full_name or
                          Path(img_name).name == img2_name_clean)
            
            matches_img3 = (img_name_clean == img3_name_clean or 
                          img_name == img3_feat_name or
                          img_name_base == img3_name_clean or
                          img_name == img3_full_name or
                          Path(img_name).name == img3_name_clean)
            
            if matches_img1:
                # Validate feature index
                if feat_idx < len(kpts1):
                    mapped_track.append((0, feat_idx))  # Image 0
            elif matches_img2:
                # Validate feature index
                if feat_idx < len(kpts2):
                    mapped_track.append((1, feat_idx))  # Image 1
            elif matches_img3:
                # Validate feature index
                if feat_idx < len(kpts3):
                    mapped_track.append((2, feat_idx))  # Image 2
        
        # Only include tracks that have at least 2 features in our 3 images
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


def create_multiple_track_visualizations(
    tracks_file: str = "outputs/tracks_cell_8928d89ac57ffff.json",
    features_file: str = "outputs/features_cache_cell_8928d89ac57ffff.json",
    matches_file: str = "outputs/matches_unfiltered_cell_8928d89ac57ffff.json",
    image_dir: str = "inputs/quarter_resolution_images",
    output_dir: str = "outputs/visualization",
    num_tracks: int = 20
):
    """
    Create visualizations for multiple tracks of different lengths.
    
    Args:
        tracks_file: Path to tracks JSON file
        features_file: Path to features cache JSON file
        image_dir: Directory containing original images
        output_dir: Directory to save visualizations
        num_tracks: Number of tracks to visualize
    """
    # Load tracks
    with open(tracks_file, 'r') as f:
        tracks_data = json.load(f)
    
    tracks = tracks_data.get('tracks', [])
    print(f"Total tracks: {len(tracks)}")
    
    if not tracks:
        print("No tracks found!")
        return
    
    # Group tracks by length
    tracks_by_length = {}
    for track in tracks:
        length = track.get('length', len(track.get('images', [])))
        if length not in tracks_by_length:
            tracks_by_length[length] = []
        tracks_by_length[length].append(track)
    
    print(f"\nTrack length distribution:")
    for length in sorted(tracks_by_length.keys()):
        print(f"  Length {length}: {len(tracks_by_length[length])} tracks")
    
    # Select tracks to visualize
    tracks_to_visualize = []
    
    # Include some length 2 tracks
    if 2 in tracks_by_length and len(tracks_by_length[2]) > 0:
        tracks_to_visualize.append(tracks_by_length[2][0])
        print(f"\nSelected length 2 track: {tracks_by_length[2][0].get('track_id', 'unknown')}")
    
    # Include some medium length tracks (3-5)
    for length in [3, 4, 5]:
        if length in tracks_by_length and len(tracks_by_length[length]) > 0:
            tracks_to_visualize.append(tracks_by_length[length][0])
            print(f"Selected length {length} track: {tracks_by_length[length][0].get('track_id', 'unknown')}")
    
    # Include some longer tracks (10+)
    long_tracks = [t for t in tracks if t.get('length', 0) >= 10]
    if long_tracks:
        # Get a few long tracks
        long_tracks_sorted = sorted(long_tracks, key=lambda t: t.get('length', 0), reverse=True)
        for i, track in enumerate(long_tracks_sorted[:min(5, len(long_tracks_sorted))]):
            if len(tracks_to_visualize) < num_tracks:
                tracks_to_visualize.append(track)
                print(f"Selected long track (length {track.get('length', 0)}): {track.get('track_id', 'unknown')}")
    
    # Fill up to num_tracks with random tracks of different lengths
    random.seed(42)  # For reproducibility
    
    # Get tracks by length that we haven't selected yet
    available_lengths = sorted([l for l in tracks_by_length.keys() if l not in [2, 3, 4, 5]])
    
    # Select random tracks from different lengths
    for length in available_lengths:
        if len(tracks_to_visualize) >= num_tracks:
            break
        available_tracks = [t for t in tracks_by_length[length] if t not in tracks_to_visualize]
        if available_tracks:
            tracks_to_visualize.append(random.choice(available_tracks))
    
    # If we still need more, fill with completely random tracks
    while len(tracks_to_visualize) < num_tracks:
        remaining = [t for t in tracks if t not in tracks_to_visualize]
        if remaining:
            tracks_to_visualize.append(random.choice(remaining))
        else:
            break
    
    print(f"\nCreating visualizations for {len(tracks_to_visualize)} tracks...")
    
    # Create visualizations
    for track in tracks_to_visualize:
        track_id = track.get('track_id')
        track_length = track.get('length', len(track.get('images', [])))
        
        # For length 2 tracks, show both images
        # For longer tracks, show first 5
        num_images_to_show = min(track_length, 5) if track_length > 2 else 2
        
        try:
            visualize_track(
                track_id=track_id,
                num_images=num_images_to_show,
                tracks_file=tracks_file,
                features_file=features_file,
                matches_file=matches_file,
                image_dir=image_dir,
                output_dir=output_dir,
                verify_matches=True  # Enable match verification to show match status
            )
            print(f"  ✓ Created visualization for track {track_id} (length {track_length})")
        except Exception as e:
            print(f"  ✗ Error visualizing track {track_id}: {e}")
    
    print(f"\n✓ Created {len(tracks_to_visualize)} track visualizations in {output_dir}/")


def visualize_overlap_counts(
    overlaps_file: str = "outputs/footprint_overlaps.json",
    min_overlap_threshold: float = 50.0,
    output_file: str = "outputs/overlap_counts_per_image.png"
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


def visualize_camera_poses(poses_file: str = "outputs/camera_poses.json",
                           output_file: str = "outputs/visualization/camera_poses.png"):
    """
    Visualize camera positions and orientations.
    
    Args:
        poses_file: Path to JSON file with camera poses
        output_file: Path to save visualization
    """
    import re
    
    # Load poses
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    # Filter to only poses with GPS
    poses_with_gps = [p for p in poses if p.get('gps') is not None]
    
    if not poses_with_gps:
        print("No GPS data available for visualization")
        return
    
    print(f"Visualizing {len(poses_with_gps)} camera poses")
    
    def parse_image_number(image_name: str) -> int:
        """Extract image number from filename (e.g., 0001 from ..._0001.jpg)."""
        match = re.search(r'_(\d{4})\.jpg$', image_name)
        if match:
            return int(match.group(1))
        return -1
    
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


def visualize_footprints_2d(
    camera_poses_file: str = "outputs/camera_poses.json",
    output_file: str = "outputs/visualization/footprints_2d.png",
    focal_length_mm: float = 8.8,
    sensor_width_mm: float = 13.2,
    sensor_height_mm: float = 9.9,
    show_image_numbers: bool = True,
    alpha: float = 0.3,
    filter_pitch: Optional[float] = None,
    filter_pitch_tolerance: float = 1.0,
    filter_oblique_only: bool = False
):
    """
    Create a 2D visualization of image footprints.
    
    Args:
        camera_poses_file: Path to camera_poses.json
        output_file: Path to output PNG file
        focal_length_mm: Focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        show_image_numbers: Whether to show image numbers on footprints
        alpha: Transparency of footprint polygons (0-1)
        filter_pitch: If provided, only show footprints with this pitch angle (degrees)
        filter_pitch_tolerance: Tolerance for pitch filtering (degrees)
        filter_oblique_only: If True, only show oblique footprints (exclude nadir, pitch != -90°)
    """
    import matplotlib.patches as patches
    from matcher.utils import compute_footprint_polygon
    
    # Load camera poses
    with open(camera_poses_file, 'r') as f:
        poses = json.load(f)
    
    # Find origin from first pose with GPS
    origin_lat = None
    origin_lon = None
    for pose in poses:
        if pose.get('gps'):
            origin_lat = pose['gps']['latitude']
            origin_lon = pose['gps']['longitude']
            break
    
    if origin_lat is None:
        raise ValueError("No GPS data found in camera poses")
    
    # Compute all footprints
    footprints = []
    image_names = []
    
    for pose in poses:
        if not pose.get('gps'):
            continue
        # Use dji_orientation if available, otherwise use default nadir orientation
        if not pose.get('dji_orientation'):
            # Default to nadir for images without orientation data
            pose['dji_orientation'] = {'gimbal_pitch': -90.0, 'gimbal_roll': 0.0, 'gimbal_yaw': 0.0}
        
        # Filter by pitch if requested
        pitch = pose['dji_orientation'].get('gimbal_pitch', -90.0)
        if filter_oblique_only:
            # Only show oblique (exclude nadir, pitch != -90°)
            if abs(pitch + 90.0) < filter_pitch_tolerance:
                continue
        elif filter_pitch is not None:
            # Filter to specific pitch angle
            if abs(pitch - filter_pitch) > filter_pitch_tolerance:
                continue
        
        footprint_poly, _ = compute_footprint_polygon(
            gps=pose['gps'],
            dji_orientation=pose['dji_orientation'],
            focal_length_mm=focal_length_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            origin_lat=origin_lat,
            origin_lon=origin_lon
        )
        
        if footprint_poly is not None:
            # Convert Shapely polygon to numpy array
            try:
                from shapely.geometry import Polygon
                if isinstance(footprint_poly, Polygon):
                    corners = np.array(footprint_poly.exterior.coords[:-1])  # Exclude closing point
                else:
                    corners = np.array(footprint_poly)
                
                # Accept footprints with 3 or more corners (trapezoids for oblique cameras)
                if len(corners) >= 3:
                    # If we have exactly 3 corners, duplicate the last one to make 4
                    if len(corners) == 3:
                        corners = np.vstack([corners, corners[-1]])
                    footprints.append(corners[:4])
                    image_names.append(pose['image_name'])
            except Exception as e:
                # If conversion fails, skip this footprint
                continue
    
    if not footprints:
        raise ValueError("No valid footprints computed")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    # Determine plot bounds
    all_x = []
    all_y = []
    for fp in footprints:
        all_x.extend(fp[:, 0])
        all_y.extend(fp[:, 1])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.05
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters, relative to origin)', fontsize=12)
    ax.set_ylabel('Y (meters, relative to origin)', fontsize=12)
    
    # Set title based on filter
    if filter_oblique_only:
        title = 'Oblique Image Footprints (2D View)\nTrapezoidal footprints for oblique angles'
    elif filter_pitch is not None:
        title = f'Image Footprints at {filter_pitch}° Pitch (2D View)'
    else:
        title = 'Image Footprints (2D View)\nShowing overlap and rotation'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw footprints with color gradient
    num_footprints = len(footprints)
    colors = plt.cm.viridis(np.linspace(0, 1, num_footprints))
    
    for i, (footprint, img_name) in enumerate(zip(footprints, image_names)):
        # Create polygon patch
        polygon = patches.Polygon(
            footprint,
            closed=True,
            edgecolor='black',
            linewidth=0.5,
            facecolor=colors[i],
            alpha=alpha
        )
        ax.add_patch(polygon)
        
        # Add image number label at centroid
        if show_image_numbers:
            centroid = footprint.mean(axis=0)
            # Extract image number from filename (e.g., "0001" from "8928d89ac57ffff_172550_0001.jpg")
            try:
                parts = img_name.split('_')
                if len(parts) >= 3:
                    img_num = parts[-1].replace('.jpg', '')
                    ax.text(centroid[0], centroid[1], img_num, 
                           fontsize=6, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
            except:
                pass
    
    # Add legend showing color progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_footprints-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Image Sequence')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved footprint visualization to: {output_file}")
    print(f"  Total footprints: {num_footprints}")
    print(f"  X range: [{x_min:.1f}, {x_max:.1f}] meters")
    print(f"  Y range: [{y_min:.1f}, {y_max:.1f}] meters")
    
    # Analyze rotation
    print("\nAnalyzing footprint rotation:")
    rotations = []
    for i, footprint in enumerate(footprints[:20]):  # Check first 20
        # Calculate angle of first edge (from corner 0 to corner 1)
        edge = footprint[1] - footprint[0]
        angle_rad = np.arctan2(edge[1], edge[0])
        angle_deg = np.degrees(angle_rad)
        rotations.append(angle_deg)
        if i < 5:
            print(f"  Image {i+1}: rotation = {angle_deg:.1f}°")
    
    if rotations:
        rotation_std = np.std(rotations)
        print(f"  Rotation std (first 20): {rotation_std:.1f}°")
        if rotation_std < 1.0:
            print("  ⚠️  Warning: Very low rotation variation - footprints may be grid-aligned")
        else:
            print("  ✓ Good rotation variation detected")
