"""
Create visualizations for multiple tracks, including length 2 tracks.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from visualize_single_track import visualize_track


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
    import random
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
                verify_matches=False  # Disable match verification for speed
            )
            print(f"  ✓ Created visualization for track {track_id} (length {track_length})")
        except Exception as e:
            print(f"  ✗ Error visualizing track {track_id}: {e}")
    
    print(f"\n✓ Created {len(tracks_to_visualize)} track visualizations in {output_dir}/")


if __name__ == "__main__":
    create_multiple_track_visualizations()
