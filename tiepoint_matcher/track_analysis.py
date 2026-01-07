"""
Track analysis utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict


def compute_tracks(matches: List[Dict], features: Dict, max_track_length: Optional[int] = None) -> Dict:
    """
    Compute feature tracks from matches.
    A track represents a single 3D point observed across multiple images.
    Each track should have at most one feature per image.
    
    CRITICAL CONSTRAINT: Tracks are built only from DIRECT matches between consecutive images.
    For a track A->B->C to be valid:
    - There must be a direct match between A and B
    - There must be a direct match between B and C
    - The feature in A matches to a feature in B, and that same feature in B matches to a feature in C
    
    The maximum track length is limited by the maximum number of overlapping images,
    since a feature can only appear in images that overlap with each other.
    
    Args:
        matches: List of match dictionaries with 'image0', 'image1', 'matches'
        features: Dictionary mapping image_path to feature dict
        max_track_length: Maximum track length (should be <= max overlap count)
        
    Returns:
        Dictionary with track information
    """
    # Build match graph: (image_name, feature_idx) -> set of (image_name, feature_idx)
    feature_matches = defaultdict(set)
    
    # Build image-to-image match lookup for fast validation
    # (img0, img1) -> {feat0_idx: feat1_idx}
    image_pair_matches = defaultdict(dict)
    
    # Build match graph
    for match_dict in matches:
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        img0_name = Path(img0_path).name
        img1_name = Path(img1_path).name
        match_indices = match_dict['matches']
        
        # Convert to numpy array if it's a list
        if isinstance(match_indices, list):
            match_indices = np.array(match_indices)
        
        # Build image pair match lookup (forward direction)
        pair_key = (img0_name, img1_name)
        pair_matches = {}
        for match in match_indices:
            idx0, idx1 = int(match[0]), int(match[1])
            pair_matches[idx0] = idx1
            feat0_id = (img0_name, idx0)
            feat1_id = (img1_name, idx1)
            
            feature_matches[feat0_id].add(feat1_id)
            feature_matches[feat1_id].add(feat0_id)
        
        image_pair_matches[pair_key] = pair_matches
        # Also store reverse direction
        reverse_pair_matches = {v: k for k, v in pair_matches.items()}
        image_pair_matches[(img1_name, img0_name)] = reverse_pair_matches
    
    # Find connected components (tracks) with strict constraints:
    # 1. One feature per image
    # 2. Consecutive images in track MUST have direct matches
    visited = set()
    tracks = []
    track_id = 0
    
    def get_matched_feature(img0: str, img1: str, feat0_idx: int) -> Optional[int]:
        """
        Get the feature index in img1 that matches with feat0_idx in img0.
        Returns None if no match exists.
        """
        key1 = (img0, img1)
        key2 = (img1, img0)
        
        if key1 in image_pair_matches:
            if feat0_idx in image_pair_matches[key1]:
                return image_pair_matches[key1][feat0_idx]
        elif key2 in image_pair_matches:
            # Reverse direction: find feat1_idx such that feat1_idx -> feat0_idx
            reverse_matches = image_pair_matches[key2]
            for feat1_idx, matched_feat0_idx in reverse_matches.items():
                if matched_feat0_idx == feat0_idx:
                    return feat1_idx
        
        return None
    
    def build_track_from_feature(start_feature_id: Tuple[str, int], max_length: int) -> List[Tuple[str, int]]:
        """
        Build a track starting from a feature, ensuring:
        1. Only one feature per image
        2. Consecutive images have direct matches (strict requirement)
        3. Track length doesn't exceed max_length
        """
        if start_feature_id in visited:
            return []
        
        track = [start_feature_id]
        visited.add(start_feature_id)
        track_images = {start_feature_id[0]}
        
        # Build track by following direct matches only
        # At each step, we can only add an image that has a direct match with the LAST image in the track
        while len(track) < max_length:
            current_img, current_feat_idx = track[-1]
            
            # Find all candidate next images that have direct matches with current image
            candidate_next = []
            for neighbor_feat_id in feature_matches[(current_img, current_feat_idx)]:
                neighbor_img, neighbor_feat_idx = neighbor_feat_id
                
                # Skip if we already have this image in the track
                if neighbor_img in track_images:
                    continue
                
                # Verify this is a direct match
                matched_idx = get_matched_feature(current_img, neighbor_img, current_feat_idx)
                if matched_idx is not None and matched_idx == neighbor_feat_idx:
                    candidate_next.append(neighbor_feat_id)
            
            if not candidate_next:
                break
            
            # Choose the first candidate (greedy approach)
            # TODO: Could be improved with better heuristics (e.g., choose based on match confidence)
            next_feat_id = candidate_next[0]
            track.append(next_feat_id)
            visited.add(next_feat_id)
            track_images.add(next_feat_id[0])
        
        return track
    
    # Build tracks starting from each unvisited feature
    for img_path, feat_dict in features.items():
        img_name = Path(img_path).name
        num_features = len(feat_dict['keypoints'])
        
        for idx in range(num_features):
            feature_id = (img_name, idx)
            if feature_id not in visited:
                track_features = build_track_from_feature(
                    feature_id, 
                    max_track_length if max_track_length else 50
                )
                
                # Only keep tracks with at least 2 features (observed in at least 2 images)
                if len(track_features) >= 2:
                    track_images = {img for img, _ in track_features}
                    tracks.append({
                        'track_id': track_id,
                        'features': track_features,
                        'length': len(track_features),  # Number of features = number of images
                        'images': track_images
                    })
                    track_id += 1
    
    return {
        'tracks': tracks,
        'num_tracks': len(tracks),
        'total_features_in_tracks': sum(len(t['features']) for t in tracks)
    }


def plot_track_length_histogram(tracks_data: Dict, output_path: str):
    """
    Create histogram of track lengths.
    
    Args:
        tracks_data: Dictionary from compute_tracks
        output_path: Path to save histogram
    """
    track_lengths = [t['length'] for t in tracks_data['tracks']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(track_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Track Length (number of features)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Feature Track Length Distribution\nTotal tracks: {tracks_data["num_tracks"]}')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    if track_lengths:
        mean_len = np.mean(track_lengths)
        median_len = np.median(track_lengths)
        max_len = max(track_lengths)
        ax.axvline(mean_len, color='r', linestyle='--', label=f'Mean: {mean_len:.1f}')
        ax.axvline(median_len, color='g', linestyle='--', label=f'Median: {median_len:.1f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved track length histogram to: {output_path}")
