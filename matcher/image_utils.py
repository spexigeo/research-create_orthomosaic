"""
Image utilities for downsampling and caching.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, Tuple, Optional


def downsample_image(image_path: str, output_path: str, scale: float = 0.25) -> str:
    """
    Downsample an image and save to output path.
    
    Args:
        image_path: Path to input image
        output_path: Path to save downsampled image
        scale: Downsampling scale (0.25 = quarter resolution)
        
    Returns:
        Path to saved downsampled image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Calculate new dimensions
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Downsample
    img_downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Save
    cv2.imwrite(str(output_path), img_downsampled)
    
    return str(output_path)


def downsample_images_batch(image_paths: list, output_dir: str, scale: float = 0.25, 
                            force_recompute: bool = False) -> Dict[str, str]:
    """
    Downsample multiple images and save to output directory.
    Skips images that already exist unless force_recompute is True.
    
    Args:
        image_paths: List of input image paths
        output_dir: Directory to save downsampled images
        scale: Downsampling scale
        force_recompute: If True, recompute even if output exists
        
    Returns:
        Dictionary mapping original image paths to downsampled image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {}
    skipped = 0
    created = 0
    
    for image_path in image_paths:
        image_name = Path(image_path).name
        output_path = output_dir / f"quarter_{image_name}"
        
        # Check if already exists
        if not force_recompute and output_path.exists():
            mapping[image_path] = str(output_path)
            skipped += 1
        else:
            downsample_image(image_path, str(output_path), scale)
            mapping[image_path] = str(output_path)
            created += 1
    
    if skipped > 0:
        print(f"   Skipped {skipped} existing quarter-resolution images")
    if created > 0:
        print(f"   Created {created} new quarter-resolution images")
    
    return mapping


def save_feature_coordinates(features: Dict, output_path: str):
    """
    Save feature coordinates to JSON file.
    
    Args:
        features: Dictionary mapping image_path to feature dict
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for image_path, feat_dict in features.items():
        image_name = Path(image_path).name
        data[image_name] = {
            'image_path': image_path,
            'keypoints': feat_dict['keypoints'].tolist(),
            'scores': feat_dict['scores'].tolist(),
            'num_features': len(feat_dict['keypoints'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Saved feature coordinates for {len(data)} images to {output_path}")


def save_features_full(features: Dict, output_path: str):
    """
    Save full feature data including descriptors (for reloading).
    Note: This creates a larger file but allows full reconstruction.
    
    Args:
        features: Dictionary mapping image_path to feature dict
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for image_path, feat_dict in features.items():
        image_name = Path(image_path).name
        data[image_name] = {
            'image_path': image_path,
            'keypoints': feat_dict['keypoints'].tolist(),
            'scores': feat_dict['scores'].tolist(),
            'descriptors': feat_dict.get('descriptors', []).tolist() if 'descriptors' in feat_dict else [],
            'num_features': len(feat_dict['keypoints'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Saved full feature data for {len(data)} images to {output_path}")


def load_features_full(input_path: str) -> Dict:
    """
    Load full feature data from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary mapping image_name to feature data (ready for use)
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    features = {}
    for image_name, feat_data in data.items():
        features[feat_data['image_path']] = {
            'keypoints': np.array(feat_data['keypoints']),
            'scores': np.array(feat_data['scores']),
            'descriptors': np.array(feat_data['descriptors']) if feat_data.get('descriptors') else None,
            'image_path': feat_data['image_path']
        }
    
    print(f"   Loaded feature data for {len(features)} images from {input_path}")
    return features


def load_feature_coordinates(input_path: str) -> Dict:
    """
    Load feature coordinates from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary mapping image_name to feature data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    for image_name in data:
        data[image_name]['keypoints'] = np.array(data[image_name]['keypoints'])
        data[image_name]['scores'] = np.array(data[image_name]['scores'])
    
    return data
