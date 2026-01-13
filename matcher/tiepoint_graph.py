"""
Tiepoint Graph: A graph-based data structure for managing features, matches, and tracks
across multiple H3 cells.

The graph tracks:
- Features (nodes): Keypoints in images
- Matches (edges): Correspondences between features
- Tracks (connected components): Sequences of matched features across images
- Cell associations: Which cell(s) each feature/match/track belongs to
"""

import json
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np


class TiepointGraph:
    """
    Graph-based data structure for managing tiepoints across multiple cells.
    
    Structure:
    - Features: (image_name, feature_idx) -> feature_data
    - Matches: (img0, feat0_idx, img1, feat1_idx) -> match_data
    - Tracks: track_id -> track_data
    - Cell associations: cell_id -> {features, matches, tracks}
    """
    
    def __init__(self):
        # Core data structures
        self.features: Dict[Tuple[str, int], Dict] = {}  # (image_name, feature_idx) -> feature_data
        self.matches: Dict[Tuple[str, int, str, int], Dict] = {}  # (img0, feat0_idx, img1, feat1_idx) -> match_data
        self.tracks: Dict[int, Dict] = {}  # track_id -> track_data
        
        # Cell associations
        self.cell_features: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)  # cell_id -> set of (image_name, feature_idx)
        self.cell_matches: Dict[str, Set[Tuple[str, int, str, int]]] = defaultdict(set)  # cell_id -> set of match keys
        self.cell_tracks: Dict[str, Set[int]] = defaultdict(set)  # cell_id -> set of track_ids
        
        # Inter-cell relationships
        self.inter_cell_matches: Dict[Tuple[str, str], Set[Tuple[str, int, str, int]]] = defaultdict(set)  # (cell1, cell2) -> matches
        
        # Metadata
        self.next_track_id = 0
        self.cell_images: Dict[str, List[str]] = {}  # cell_id -> list of image paths
        
    def add_feature(self, image_name: str, feature_idx: int, feature_data: Dict, cell_id: str):
        """Add a feature to the graph."""
        key = (image_name, feature_idx)
        self.features[key] = feature_data
        self.cell_features[cell_id].add(key)
    
    def add_match(self, img0: str, feat0_idx: int, img1: str, feat1_idx: int, 
                  match_data: Dict, cell0: str, cell1: str):
        """Add a match to the graph."""
        # Normalize order (smaller image name first, then smaller feature idx)
        if img0 > img1 or (img0 == img1 and feat0_idx > feat1_idx):
            img0, feat0_idx, img1, feat1_idx = img1, feat1_idx, img0, feat0_idx
            cell0, cell1 = cell1, cell0
        
        key = (img0, feat0_idx, img1, feat1_idx)
        self.matches[key] = match_data
        
        # Add to cell associations
        if cell0 == cell1:
            # Intra-cell match
            self.cell_matches[cell0].add(key)
        else:
            # Inter-cell match
            self.cell_matches[cell0].add(key)
            self.cell_matches[cell1].add(key)
            self.inter_cell_matches[(cell0, cell1)].add(key)
            self.inter_cell_matches[(cell1, cell0)].add(key)
    
    def add_track(self, track_data: Dict, cell_ids: Set[str]):
        """Add a track to the graph."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        track_data['track_id'] = track_id
        track_data['cells'] = list(cell_ids)
        self.tracks[track_id] = track_data
        
        # Add to cell associations
        for cell_id in cell_ids:
            self.cell_tracks[cell_id].add(track_id)
        
        return track_id
    
    def get_cell_features(self, cell_id: str) -> Dict[Tuple[str, int], Dict]:
        """Get all features for a specific cell."""
        return {key: self.features[key] for key in self.cell_features[cell_id]}
    
    def get_cell_matches(self, cell_id: str) -> List[Dict]:
        """Get all matches for a specific cell (intra-cell only)."""
        matches = []
        for key in self.cell_matches[cell_id]:
            img0, feat0_idx, img1, feat1_idx = key
            # Only include if both images are in the same cell
            cell0 = self._get_cell_for_image(img0)
            cell1 = self._get_cell_for_image(img1)
            if cell0 == cell_id and cell1 == cell_id:
                matches.append({
                    'image0': img0,
                    'image1': img1,
                    'feature0_idx': feat0_idx,
                    'feature1_idx': feat1_idx,
                    **self.matches[key]
                })
        return matches
    
    def get_inter_cell_matches(self, cell1: str, cell2: str) -> List[Dict]:
        """Get all matches between two cells."""
        matches = []
        for key in self.inter_cell_matches.get((cell1, cell2), set()):
            img0, feat0_idx, img1, feat1_idx = key
            matches.append({
                'image0': img0,
                'image1': img1,
                'feature0_idx': feat0_idx,
                'feature1_idx': feat1_idx,
                **self.matches[key]
            })
        return matches
    
    def get_cell_tracks(self, cell_id: str) -> List[Dict]:
        """Get all tracks for a specific cell."""
        return [self.tracks[track_id] for track_id in self.cell_tracks[cell_id]]
    
    def _get_cell_for_image(self, image_name: str) -> Optional[str]:
        """Find which cell an image belongs to."""
        for cell_id, images in self.cell_images.items():
            for img_path in images:
                if Path(img_path).name == image_name or image_name in img_path:
                    return cell_id
        return None
    
    def link_tracks_across_cells(self) -> Dict:
        """
        Link tracks across cells to form longer tracks.
        
        Two tracks can be linked if:
        1. They share a common feature (same image, same feature index)
        2. They are in different cells or have overlapping images
        
        Returns:
            Dictionary with linked track information
        """
        # Build feature -> track mapping
        feature_to_tracks: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
        for track_id, track_data in self.tracks.items():
            for img_name, feat_idx in track_data['features']:
                feature_to_tracks[(img_name, feat_idx)].add(track_id)
        
        # Find linkable tracks (tracks that share features)
        linked_tracks = []
        processed_tracks = set()
        
        for track_id, track_data in self.tracks.items():
            if track_id in processed_tracks:
                continue
            
            # Start a new linked track group
            linked_group = {track_id}
            processed_tracks.add(track_id)
            
            # Find all tracks that share features with this track
            queue = [track_id]
            while queue:
                current_track_id = queue.pop(0)
                current_track = self.tracks[current_track_id]
                
                # Check all features in current track
                for img_name, feat_idx in current_track['features']:
                    # Find other tracks with the same feature
                    for other_track_id in feature_to_tracks[(img_name, feat_idx)]:
                        if other_track_id not in processed_tracks:
                            linked_group.add(other_track_id)
                            processed_tracks.add(other_track_id)
                            queue.append(other_track_id)
            
            # Create linked track
            if len(linked_group) > 1:
                # Merge features from all tracks in the group
                all_features = []
                all_images = set()
                all_cells = set()
                
                for tid in linked_group:
                    track = self.tracks[tid]
                    all_features.extend(track['features'])
                    all_images.update(track.get('images', set()))
                    all_cells.update(track.get('cells', []))
                
                # Remove duplicate features (same image, same feature index)
                unique_features = []
                seen_features = set()
                for feat in all_features:
                    if feat not in seen_features:
                        unique_features.append(feat)
                        seen_features.add(feat)
                
                linked_tracks.append({
                    'linked_track_id': len(linked_tracks),
                    'component_track_ids': list(linked_group),
                    'features': unique_features,
                    'length': len(unique_features),
                    'images': all_images,
                    'cells': list(all_cells)
                })
        
        return {
            'linked_tracks': linked_tracks,
            'num_linked_tracks': len(linked_tracks),
            'num_original_tracks': len(self.tracks)
        }
    
    def remove_cell(self, cell_id: str):
        """Remove all data associated with a cell (for updates)."""
        # Remove features
        for key in list(self.cell_features[cell_id]):
            del self.features[key]
        del self.cell_features[cell_id]
        
        # Remove matches
        matches_to_remove = []
        for key in self.cell_matches[cell_id]:
            img0, feat0_idx, img1, feat1_idx = key
            cell0 = self._get_cell_for_image(img0)
            cell1 = self._get_cell_for_image(img1)
            if cell0 == cell_id or cell1 == cell_id:
                matches_to_remove.append(key)
        
        for key in matches_to_remove:
            if key in self.matches:
                del self.matches[key]
        
        del self.cell_matches[cell_id]
        
        # Remove tracks
        for track_id in list(self.cell_tracks[cell_id]):
            # Only remove if track is only in this cell
            track = self.tracks[track_id]
            if set(track.get('cells', [])) == {cell_id}:
                del self.tracks[track_id]
            else:
                # Remove cell from track's cell list
                track['cells'] = [c for c in track.get('cells', []) if c != cell_id]
        
        del self.cell_tracks[cell_id]
        
        # Remove inter-cell matches
        for (c1, c2) in list(self.inter_cell_matches.keys()):
            if c1 == cell_id or c2 == cell_id:
                del self.inter_cell_matches[(c1, c2)]
    
    def save(self, output_file: str):
        """Save graph to JSON file."""
        # Convert to JSON-serializable format
        graph_data = {
            'features': {
                f"{img}_{idx}": data for (img, idx), data in self.features.items()
            },
            'matches': {
                f"{img0}_{f0}_{img1}_{f1}": data 
                for (img0, f0, img1, f1), data in self.matches.items()
            },
            'tracks': self.tracks,
            'cell_features': {
                cell_id: [f"{img}_{idx}" for (img, idx) in features]
                for cell_id, features in self.cell_features.items()
            },
            'cell_matches': {
                cell_id: [f"{img0}_{f0}_{img1}_{f1}" for (img0, f0, img1, f1) in matches]
                for cell_id, matches in self.cell_matches.items()
            },
            'cell_tracks': {
                cell_id: list(track_ids) for cell_id, track_ids in self.cell_tracks.items()
            },
            'inter_cell_matches': {
                f"{c1}_{c2}": [f"{img0}_{f0}_{img1}_{f1}" for (img0, f0, img1, f1) in matches]
                for (c1, c2), matches in self.inter_cell_matches.items()
            },
            'cell_images': self.cell_images,
            'next_track_id': self.next_track_id
        }
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load(self, input_file: str):
        """Load graph from JSON file."""
        with open(input_file, 'r') as f:
            graph_data = json.load(f)
        
        # Reconstruct features
        self.features = {}
        for key_str, data in graph_data['features'].items():
            parts = key_str.rsplit('_', 1)
            if len(parts) == 2:
                img_name = parts[0]
                feat_idx = int(parts[1])
                self.features[(img_name, feat_idx)] = data
        
        # Reconstruct matches
        self.matches = {}
        for key_str, data in graph_data['matches'].items():
            parts = key_str.split('_')
            if len(parts) >= 4:
                img0 = '_'.join(parts[:-3])
                f0 = int(parts[-3])
                img1 = '_'.join(parts[-2:-1])
                f1 = int(parts[-1])
                self.matches[(img0, f0, img1, f1)] = data
        
        # Reconstruct tracks
        self.tracks = {int(k): v for k, v in graph_data['tracks'].items()}
        
        # Reconstruct cell associations
        self.cell_features = {
            cell_id: {tuple(f.split('_', 1)) for f in features}
            for cell_id, features in graph_data.get('cell_features', {}).items()
        }
        
        self.cell_images = graph_data.get('cell_images', {})
        self.next_track_id = graph_data.get('next_track_id', len(self.tracks))
