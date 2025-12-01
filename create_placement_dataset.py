import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from detect_placement import PlacementDetector
from cr_element import crop_arena

def extract_arena_and_frame_info(image_path: Path) -> Tuple[str, int]:
    parts = image_path.parts
    arena = "unknown"
    for part in parts:
        if part.startswith("arena_"):
            arena = part.replace("arena_", "")
            break
    frame_num = 0
    stem = image_path.stem 
    if "frame_" in stem or stem.isdigit():
        try:
            frame_num = int(stem.split("_")[-1])
        except (ValueError, IndexError):
            frame_num = 0
    
    return arena, frame_num


def detect_blue_clock_placements(image_path: str, 
                                 detector: PlacementDetector,
                                 min_radius: int = 10,
                                 max_radius: int = 30) -> List[Tuple[int, int, float, str]]:
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Warning: Could not load image from {image_path}")
        return []
    img = crop_arena(raw_img)
    detections = detector.detect_placements(
        img, 
        use_all_templates=True,
        return_template_id=True
    )
    
    return detections


def process_game_directory(game_dir: Path,
                          detector: Optional[PlacementDetector] = None,
                          troop_name: str = "unknown") -> pd.DataFrame:
    """
    Process all frames in a game directory and detect placements
    """
    images_dir = game_dir / "images"
    if not images_dir.exists():
        print(f"No images directory found in {game_dir}")
        return pd.DataFrame(columns=['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id'])

    game_id = game_dir.name
    
    arena, _ = extract_arena_and_frame_info(images_dir / "dummy.png")
    
    all_placements = []
    
    image_files = sorted(images_dir.glob("*.png"))
    print(f"Processing {len(image_files)} frames from {game_dir.name}...")
    
    for img_path in image_files:
        _, frame_num = extract_arena_and_frame_info(img_path)
        
        detections = detect_blue_clock_placements(str(img_path), detector)
        #add to results
        for x, y, confidence, template_id in detections:
            all_placements.append({
                'troop': troop_name,
                'x': x,
                'y': y,
                'arena': arena,
                'frame': frame_num,
                'game_id': game_id,
                'template_id': template_id,
                'confidence': confidence
            })
    
    df = pd.DataFrame(all_placements)
    
    if len(df) > 0:
        print(f"  Found {len(df)} placements across {df['frame'].nunique()} frames")
    else:
        print(f"  No placements detected")
    
    return df


def process_multiple_games(data_root: Path,
                          arenas: List[str] = None,
                          output_csv: str = "troop_placements.csv",
                          template_dir: Optional[str] = None,
                          game_filter: Optional[str] = None) -> pd.DataFrame:
    detector = None
    detector = PlacementDetector(threshold=0.65)
    detector.load_multiple_templates(template_dir)
    
    #If no arenas specified, find all arena directories
    if arenas is None:
        arenas = [d.name for d in data_root.iterdir() 
                 if d.is_dir() and d.name.startswith("arena_")]
        arenas = sorted(arenas)
    
    print(f"\n{'='*70}")
    print(f"TROOP PLACEMENT DETECTION")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"Arenas to process: {len(arenas)}")
    
    all_data = []
    
    for arena_name in arenas:
        arena_dir = data_root / arena_name
        if not arena_dir.exists():
            continue
        print(f"\n[{arena_name}]")
        game_dirs = sorted([d for d in arena_dir.iterdir() 
                          if d.is_dir() and (d.name.startswith("game_") or 
                                            len(d.name) == 36 or  
                                            (d / "images").exists())])  
        if game_filter:
            game_dirs = [d for d in game_dirs if game_filter in d.name]
        
        if not game_dirs:
            msg = f"  No game directories found in {arena_name}"
            if game_filter:
                msg += f" matching '{game_filter}'"
            print(msg)
            continue
        
        print(f"  Found {len(game_dirs)} game directories")
        
        for game_dir in game_dirs:
            print(f"  {game_dir.name}:")
            df = process_game_directory(
                game_dir,
                detector=detector,
            )
            
            if len(df) > 0:
                all_data.append(df)
    
    #Combine data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['arena', 'game_id', 'frame'])
        output_df = combined_df[['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id']]
        output_df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"Total placements detected: {len(combined_df)}")
        print(f"Unique games processed: {combined_df['game_id'].nunique()}")
        print(f"Arenas covered: {combined_df['arena'].nunique()}")
        print(f"Frames with placements: {combined_df['frame'].nunique()}")
        print(f"Templates used: {combined_df['template_id'].nunique()}")
        print(f"\nOutput saved to: {output_csv}")
        print(f"Columns: troop, x, y, arena, frame, game_id, template_id")
        print(f"{'='*70}\n")
        print("Sample data:")
        print(output_df.head(10))
        
        return combined_df
    else:
        print("\nNo placements detected in any games!")
        return pd.DataFrame(columns=['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id'])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect troop placements and create training dataset"
    )
    parser.add_argument("data_dir", help="root directory containing arena folders")
    parser.add_argument("--output", "-o", default="troop_placements.csv",
                       help="Output CSV file (default: troop_placements.csv)")
    parser.add_argument("--templates", "-t", type=str, required=True,
                       help="template directory")
    parser.add_argument("--arenas", "-a", nargs="+",
                       help="arenas to process")
    parser.add_argument("--game", "-g", type=str, default=None,
                       help="game to process")
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        print(f"Error: Data directory not found: {data_root}")
        return
    #Process all games
    df = process_multiple_games(
        data_root=data_root,
        arenas=args.arenas,
        output_csv=args.output,
        template_dir=args.templates,
        game_filter=args.game
    )

if __name__ == "__main__":
    main()
