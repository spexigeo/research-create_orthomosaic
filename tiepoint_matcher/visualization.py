"""
Visualization functions for features and matches.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import cv2


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
        
        # Load images
        img0 = np.array(Image.open(img0_path))
        img1 = np.array(Image.open(img1_path))
        
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
