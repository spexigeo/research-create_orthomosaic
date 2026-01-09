"""
Export tiepoints to MetaShape-compatible format.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np


def _to_native_type(value):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def export_to_metashape(match_results: Dict, output_path: str, 
                       image_base_path: Optional[str] = None) -> str:
    """
    Export features and matches to MetaShape-compatible JSON format.
    
    MetaShape format:
    {
        "cameras": {
            "image1.jpg": {
                "path": "path/to/image1.jpg",
                "sensor": {"width": <width>, "height": <height>, "type": "frame"},  # Dimensions read from actual image
                "features": [
                    {"id": 0, "x": 100.5, "y": 200.3},
                    ...
                ]
            },
            ...
        },
        "matches": [
            {
                "image1": "image1.jpg",
                "image2": "image2.jpg",
                "matches": [
                    {"feature1": 0, "feature2": 5},
                    ...
                ]
            },
            ...
        ]
    }
    
    Args:
        match_results: Dictionary from LightGlueMatcher.match_cell_images()
        output_path: Path to output JSON file
        image_base_path: Optional base path to prepend to image filenames
        
    Returns:
        Path to output file
    """
    from PIL import Image
    
    features = match_results['features']
    matches = match_results['matches']
    
    # Build cameras dictionary
    cameras = {}
    for image_path, feat_dict in features.items():
        # Get image filename
        image_filename = Path(image_path).name
        
        # Get image dimensions - must be readable from image
        img = Image.open(image_path)
        width, height = img.size
        
        # Build features list with local indices
        camera_features = []
        feature_id_map = {}  # Map from keypoint index to local feature index
        
        keypoints = feat_dict['keypoints']
        for idx, kp in enumerate(keypoints):
            feature_id_map[idx] = len(camera_features)
            camera_features.append({
                "id": len(camera_features),
                "x": _to_native_type(kp[0]),
                "y": _to_native_type(kp[1])
            })
        
        # Build full path
        if image_base_path:
            full_path = str(Path(image_base_path) / image_filename)
        else:
            full_path = image_path
        
        cameras[image_filename] = {
            "path": full_path,
            "sensor": {
                "width": int(width),
                "height": int(height),
                "type": "frame"
            },
            "features": camera_features,
            "_feature_id_map": feature_id_map  # Internal mapping for matches
        }
    
    # Build matches list
    matches_list = []
    for match_dict in matches:
        img0_path = match_dict['image0']
        img1_path = match_dict['image1']
        match_indices = match_dict['matches']  # [M, 2] array or list
        match_confidence = match_dict.get('match_confidence', None)
        
        # Convert to numpy array if it's a list
        if isinstance(match_indices, list):
            match_indices = np.array(match_indices)
        if match_confidence is not None and isinstance(match_confidence, list):
            match_confidence = np.array(match_confidence)
        
        img0_filename = Path(img0_path).name
        img1_filename = Path(img1_path).name
        
        # Get feature ID maps
        map0 = cameras[img0_filename]["_feature_id_map"]
        map1 = cameras[img1_filename]["_feature_id_map"]
        
        # Convert matches to local feature indices
        match_list = []
        for i, match_idx in enumerate(match_indices):
            idx0, idx1 = int(match_idx[0]), int(match_idx[1])
            
            # Map to local feature indices
            if idx0 in map0 and idx1 in map1:
                local_feat0 = map0[idx0]
                local_feat1 = map1[idx1]
                
                match_entry = {
                    "feature1": local_feat0,
                    "feature2": local_feat1
                }
                
                # Add confidence if available
                if match_confidence is not None and len(match_confidence) > i:
                    match_entry["confidence"] = _to_native_type(
                        match_confidence[i]
                    )
                
                match_list.append(match_entry)
        
        if match_list:
            matches_list.append({
                "image1": img0_filename,
                "image2": img1_filename,
                "matches": match_list
            })
    
    # Remove internal mapping before export
    for camera in cameras.values():
        camera.pop("_feature_id_map", None)
    
    # Create output structure
    output_data = {
        "cameras": cameras,
        "matches": matches_list,
        "metadata": {
            "num_cameras": int(len(cameras)),
            "num_match_pairs": int(len(matches_list)),
            "total_match_pairs": int(sum(len(m["matches"]) for m in matches_list))
        }
    }
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Exported {len(cameras)} cameras and {len(matches_list)} match pairs to {output_path}")
    return str(output_path)
