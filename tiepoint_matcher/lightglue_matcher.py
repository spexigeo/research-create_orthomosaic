"""
LightGlue-based tiepoint matcher for feature detection and matching.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import torchvision.transforms.functional as TF
import lightglue
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd


class LightGlueMatcher:
    """
    LightGlue-based matcher for finding tiepoints within and across H3 cells.
    """
    
    def __init__(self, extractor_type: str = 'superpoint', device: Optional[str] = None,
                 max_features_per_image: Optional[int] = None):
        """
        Initialize the LightGlue matcher.
        
        Args:
            extractor_type: Type of feature extractor ('superpoint', 'disk', 'sift', 'aliked')
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_features_per_image: Maximum number of features to keep per image (None = no limit)
        """
        if device is None:
            # Try CUDA first, then MPS (Apple Silicon), then CPU
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.max_features_per_image = max_features_per_image
        
        # Initialize extractor
        if extractor_type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        elif extractor_type == 'disk':
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        elif extractor_type == 'sift':
            self.extractor = SIFT(max_num_keypoints=2048).eval().to(self.device)
        elif extractor_type == 'aliked':
            self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        # Initialize matcher
        self.matcher = LightGlue(features=extractor_type).eval().to(self.device)
        
        self.extractor_type = extractor_type
    
    def extract_features(self, image_path: str) -> Dict:
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with keys: 'keypoints', 'descriptors', 'scores', 
            'keypoints_tensor', 'descriptors_tensor', 'scores_tensor', 'image_tensor'
        """
        # Load image - SuperPoint expects grayscale (1 channel)
        # For SuperPoint, load as grayscale directly
        if self.extractor_type == 'superpoint':
            # Load image and convert to grayscale
            pil_image = Image.open(image_path).convert('L')  # Convert to grayscale
            # Convert to tensor: TF.to_tensor gives [1, H, W] for grayscale, we need [1, 1, H, W]
            image = TF.to_tensor(pil_image).unsqueeze(0)  # Add batch dimension: [1, 1, H, W]
            # Verify shape
            if image.shape[1] != 1:
                raise ValueError(f"Expected 1 channel for SuperPoint, got {image.shape[1]}. Shape: {image.shape}")
            image_tensor = image.to(self.device)
        else:
            # For other extractors, use load_image (returns RGB)
            image = load_image(image_path)
            image_tensor = image.to(self.device)
        
        with torch.no_grad():
            feats = self.extractor({'image': image_tensor})
        
        # Store tensors (for efficient matching)
        keypoints_tensor = feats['keypoints']
        descriptors_tensor = feats['descriptors']
        scores_tensor = feats['keypoint_scores']
        
        # Limit number of features if max_features_per_image is set
        if self.max_features_per_image is not None:
            num_features = keypoints_tensor.shape[1]  # [1, N, 2]
            if num_features > self.max_features_per_image:
                # Sort by scores (descending) and take top N
                scores_flat = scores_tensor[0]  # [N]
                top_indices = torch.argsort(scores_flat, descending=True)[:self.max_features_per_image]
                top_indices = top_indices.sort()[0]  # Keep original order for consistency
                
                # Select top features
                keypoints_tensor = keypoints_tensor[:, top_indices, :]
                descriptors_tensor = descriptors_tensor[:, top_indices, :]
                scores_tensor = scores_tensor[:, top_indices]
        
        # Convert to numpy for export/visualization
        keypoints = keypoints_tensor.cpu().numpy()[0]  # [N, 2]
        descriptors = descriptors_tensor.cpu().numpy()[0]  # [N, D]
        scores = scores_tensor.cpu().numpy()[0]  # [N]
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
            'keypoints_tensor': keypoints_tensor,
            'descriptors_tensor': descriptors_tensor,
            'scores_tensor': scores_tensor,
            'image_tensor': image_tensor,
            'image_path': image_path
        }
    
    def match_features(self, feats0: Dict, feats1: Dict) -> Dict:
        """
        Match features between two images.
        
        Args:
            feats0: Features from first image (must have tensor versions)
            feats1: Features from second image (must have tensor versions)
            
        Returns:
            Dictionary with keys: 'matches', 'match_confidence'
        """
        # Ensure tensors are available and are actually tensors
        if 'keypoints_tensor' not in feats0 or not isinstance(feats0['keypoints_tensor'], torch.Tensor):
            # Re-extract if tensors are missing or wrong type
            if self.extractor_type == 'superpoint':
                pil_image0 = Image.open(feats0['image_path']).convert('L')
                image0 = TF.to_tensor(pil_image0).unsqueeze(0).to(self.device)
            else:
                image0 = load_image(feats0['image_path']).to(self.device)
            with torch.no_grad():
                feats0_tensor = self.extractor({'image': image0})
            feats0['keypoints_tensor'] = feats0_tensor['keypoints']
            feats0['descriptors_tensor'] = feats0_tensor['descriptors']
            feats0['scores_tensor'] = feats0_tensor['keypoint_scores']
            feats0['image_tensor'] = image0
        
        if 'keypoints_tensor' not in feats1 or not isinstance(feats1['keypoints_tensor'], torch.Tensor):
            # Re-extract if tensors are missing or wrong type
            if self.extractor_type == 'superpoint':
                pil_image1 = Image.open(feats1['image_path']).convert('L')
                image1 = TF.to_tensor(pil_image1).unsqueeze(0).to(self.device)
            else:
                image1 = load_image(feats1['image_path']).to(self.device)
            with torch.no_grad():
                feats1_tensor = self.extractor({'image': image1})
            feats1['keypoints_tensor'] = feats1_tensor['keypoints']
            feats1['descriptors_tensor'] = feats1_tensor['descriptors']
            feats1['scores_tensor'] = feats1_tensor['keypoint_scores']
            feats1['image_tensor'] = image1
        
        # Ensure tensors are on the correct device
        feats0['keypoints_tensor'] = feats0['keypoints_tensor'].to(self.device)
        feats0['descriptors_tensor'] = feats0['descriptors_tensor'].to(self.device)
        feats1['keypoints_tensor'] = feats1['keypoints_tensor'].to(self.device)
        feats1['descriptors_tensor'] = feats1['descriptors_tensor'].to(self.device)
        
        # Get image size from tensor
        image0_size = feats0['image_tensor'].shape[-2:][::-1]  # (width, height)
        image1_size = feats1['image_tensor'].shape[-2:][::-1]
        
        # Match features
        with torch.no_grad():
            pred = self.matcher({
                'image0': {'keypoints': feats0['keypoints_tensor'],
                          'descriptors': feats0['descriptors_tensor'],
                          'image_size': image0_size},
                'image1': {'keypoints': feats1['keypoints_tensor'],
                          'descriptors': feats1['descriptors_tensor'],
                          'image_size': image1_size}
            })
        
        # Extract matches - LightGlue returns matches as tensor with shape [1, M, 2]
        matches_tensor = pred['matches']
        
        # Handle different return formats from LightGlue
        if isinstance(matches_tensor, torch.Tensor):
            # Move to CPU before converting to numpy (required for MPS tensors)
            matches = matches_tensor.cpu().numpy()
            if matches.ndim == 3:  # [1, M, 2]
                matches = matches[0]
            elif matches.ndim == 2:  # [M, 2]
                matches = matches
            else:
                matches = np.array([]).reshape(0, 2)
        elif isinstance(matches_tensor, list):
            # Handle list - might contain tensors
            if len(matches_tensor) > 0:
                # Check if first element is a tensor
                if isinstance(matches_tensor[0], torch.Tensor):
                    # Convert list of tensors to numpy
                    matches = torch.stack(matches_tensor).cpu().numpy() if len(matches_tensor) > 1 else matches_tensor[0].cpu().numpy()
                else:
                    matches = np.array(matches_tensor)
                
                if matches.ndim == 3:  # [1, M, 2]
                    matches = matches[0]
                elif matches.ndim == 2:  # [M, 2]
                    matches = matches
                else:
                    matches = np.array([]).reshape(0, 2)
            else:
                matches = np.array([]).reshape(0, 2)
        elif isinstance(matches_tensor, np.ndarray):
            matches = matches_tensor
            if matches.ndim == 3:  # [1, M, 2]
                matches = matches[0]
            elif matches.ndim != 2:  # Not [M, 2]
                matches = np.array([]).reshape(0, 2)
        else:
            matches = np.array([]).reshape(0, 2)
        
        # Extract matching scores
        confidence_tensor = pred.get('matching_scores', None)
        if confidence_tensor is not None:
            if isinstance(confidence_tensor, torch.Tensor):
                # Move to CPU before converting to numpy (required for MPS tensors)
                match_confidence = confidence_tensor.cpu().numpy()
                if match_confidence.ndim > 1:
                    match_confidence = match_confidence[0]
            elif isinstance(confidence_tensor, list):
                # Handle list - might contain tensors
                if len(confidence_tensor) > 0 and isinstance(confidence_tensor[0], torch.Tensor):
                    match_confidence = torch.stack(confidence_tensor).cpu().numpy() if len(confidence_tensor) > 1 else confidence_tensor[0].cpu().numpy()
                else:
                    match_confidence = np.array(confidence_tensor)
                if match_confidence.ndim > 1:
                    match_confidence = match_confidence[0]
            elif isinstance(confidence_tensor, np.ndarray):
                match_confidence = confidence_tensor
                if match_confidence.ndim > 1:
                    match_confidence = match_confidence[0]
            else:
                match_confidence = np.array([])
        else:
            match_confidence = np.array([])
        
        # Filter valid matches (indices >= 0)
        if len(matches) > 0:
            valid = matches[:, 0] >= 0
            matches = matches[valid]
            # Only filter confidence if it has the same length
            if len(match_confidence) > 0 and len(match_confidence) == len(valid):
                match_confidence = match_confidence[valid]
            elif len(match_confidence) == 0:
                # Create empty array matching matches length
                match_confidence = np.array([])
        else:
            matches = np.array([]).reshape(0, 2)
            match_confidence = np.array([])
        
        return {
            'matches': matches,  # [M, 2] array of (idx0, idx1) pairs
            'match_confidence': match_confidence,
            'num_matches': len(matches)
        }
    
    def match_cell_images(self, image_paths: List[str], 
                         max_pairs: Optional[int] = None) -> Dict:
        """
        Match features across all image pairs within a cell (INTRA matches).
        
        Args:
            image_paths: List of image paths for the cell
            max_pairs: Optional limit on number of pairs to match (for testing)
            
        Returns:
            Dictionary with:
            - 'features': Dict mapping image_path to feature dict
            - 'matches': List of match dicts for each image pair
        """
        print(f"Extracting features from {len(image_paths)} images...")
        
        # Extract features from all images
        features = {}
        for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            image_name = Path(image_path).name
            print(f"\n[{idx+1}/{len(image_paths)}] Processing {image_name}...")
            feat_dict = self.extract_features(image_path)
            features[image_path] = feat_dict
            
            # Output feature coordinates for progress tracking
            num_features = len(feat_dict['keypoints'])
            print(f"  Found {num_features} features")
            if num_features > 0:
                # Show first 5 and last 5 feature coordinates
                kpts = feat_dict['keypoints']
                print(f"  First feature: ({kpts[0][0]:.1f}, {kpts[0][1]:.1f})")
                if num_features > 1:
                    print(f"  Last feature: ({kpts[-1][0]:.1f}, {kpts[-1][1]:.1f})")
                if num_features > 10:
                    sample_indices = [0, num_features//4, num_features//2, 3*num_features//4, num_features-1]
                    sample_features = [(float(kpts[i][0]), float(kpts[i][1])) for i in sample_indices if i < num_features]
                    print(f"  Sample features: {sample_features}")
        
        print(f"Matching features across {len(image_paths)} images...")
        
        # Match all pairs
        matches = []
        image_list = list(image_paths)
        
        # Generate all pairs
        pairs = []
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                pairs.append((i, j))
        
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        for i, j in tqdm(pairs, desc="Matching pairs"):
            img0_path = image_list[i]
            img1_path = image_list[j]
            
            feats0 = features[img0_path]
            feats1 = features[img1_path]
            
            match_result = self.match_features(feats0, feats1)
            
            matches.append({
                'image0': img0_path,
                'image1': img1_path,
                'matches': match_result['matches'],
                'match_confidence': match_result['match_confidence'],
                'num_matches': match_result['num_matches']
            })
        
        return {
            'features': features,
            'matches': matches
        }
