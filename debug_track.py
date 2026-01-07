"""
Debug script to verify track matches.
"""
import json
import numpy as np
from pathlib import Path

# Load data
tracks = json.load(open('outputs/tracks_cell_8928d89ac57ffff.json'))
matches = json.load(open('outputs/matches_cache_cell_8928d89ac57ffff.json'))
features = json.load(open('outputs/features_cache_cell_8928d89ac57ffff.json'))

# Get track 11
track_11 = [t for t in tracks['tracks'] if t['track_id'] == 11][0]
print(f"Track 11: {track_11['length']} features")
print(f"First 5 features: {track_11['features'][:5]}\n")

# Check matches between consecutive images
for i in range(len(track_11['features']) - 1):
    img1, idx1 = track_11['features'][i]
    img2, idx2 = track_11['features'][i + 1]
    
    print(f"Checking: {img1} feature {idx1} -> {img2} feature {idx2}")
    
    # Find match between these two images
    # Note: matches cache might use full paths, track uses image names
    img1_name = Path(img1).name if '/' in img1 else img1
    img2_name = Path(img2).name if '/' in img2 else img2
    
    found_match = False
    for match in matches:
        match_img0 = Path(match['image0']).name if '/' in match['image0'] else match['image0']
        match_img1 = Path(match['image1']).name if '/' in match['image1'] else match['image1']
        
        if (match_img0 == img1_name and match_img1 == img2_name) or \
           (match_img0 == img2_name and match_img1 == img1_name):
            matches_list = np.array(match['matches'])
            
            # Check direction
            if match_img0 == img2_name:
                # Need to swap
                matches_list = matches_list[:, [1, 0]]
            
            # Check if (idx1, idx2) is in matches
            for m in matches_list:
                if int(m[0]) == idx1 and int(m[1]) == idx2:
                    found_match = True
                    print(f"  ✓ MATCH FOUND")
                    break
            
            if not found_match:
                print(f"  ✗ NO DIRECT MATCH")
                # Show what feature idx1 matches to in img2
                matches_from_idx1 = [m for m in matches_list if int(m[0]) == idx1]
                if matches_from_idx1:
                    print(f"    Feature {idx1} in {img1} matches to: {[int(m[1]) for m in matches_from_idx1[:5]]}")
                # Show what matches to idx2 in img2
                matches_to_idx2 = [m for m in matches_list if int(m[1]) == idx2]
                if matches_to_idx2:
                    print(f"    Feature {idx2} in {img2} is matched from: {[int(m[0]) for m in matches_to_idx2[:5]]}")
            break
    
    if not found_match:
        print(f"  ✗ NO MATCH DATA FOUND between these images")
    print()
