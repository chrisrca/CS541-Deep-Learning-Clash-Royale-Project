import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import argparse


def load_detection_csv(csv_path: str) -> pd.DataFrame:
    """Load detection csv"""
    df = pd.read_csv(csv_path)
    required = ['x', 'y', 'arena', 'frame', 'game_id']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"detection csv missing columns: {missing}")
    
    print(f"Loaded {len(df)} detections from {csv_path}")
    
    return df


def find_placement_in_window(detections: pd.DataFrame, 
                             arena: str, 
                             game_id: str, 
                             frame: int,
                             window_size: int = 5) -> tuple:
    #Search for placement detection in frame window
    # Search in frame window: [frame, frame+1, ..., frame+window_size-1]
    frame_start = frame
    frame_end = frame + window_size
    arena_int = int(arena) if arena.isdigit() else int(arena.replace('arena_', ''))
    
    # Filter detections for this game, arena, and frame window
    mask = (
        (detections['arena'] == arena_int) &
        (detections['game_id'] == game_id) &
        (detections['frame'] >= frame_start) &
        (detections['frame'] < frame_end)
    )
    matches = detections[mask]
    if len(matches) == 0:
        return (-1, -1)
    matches = matches.copy()
    matches['frame_dist'] = abs(matches['frame'] - frame)
    closest = matches.sort_values('frame_dist').iloc[0]
    
    return (int(closest['x']), int(closest['y']))


def update_training_parquet(parquet_path: str,
                            detection_csv: str,
                            output_path: str,
                            window_size: int = 5):
    """
    Update training parquet with detected placement coordinates
    """    
    # Load detection results
    print("\nLoading detection CSV")
    detections = load_detection_csv(detection_csv)
    print("\nLoading training parquet")
    columns_to_read = ['card', 'x', 'y', 'arena', 'replay', 'frame']
    train_df = pd.read_parquet(parquet_path, columns=columns_to_read)
    print(f"Training data: {len(train_df)} rows")
    print(f"Columns: {list(train_df.columns)}")
    required = ['card', 'arena', 'replay', 'frame']
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ValueError(f"detection csv missing columns: {missing}")
    
    #add x,y columns if they don't exist
    if 'x' not in train_df.columns:
        train_df['x'] = -1
    if 'y' not in train_df.columns:
        train_df['y'] = 1
    group_cols = ['card', 'arena', 'replay', 'frame']
    # Process each row
    print("\nMatching detections to training data")
    matched_count = 0
    unmatched_count = 0
    for group_keys, group in train_df.groupby(group_cols):
        frame, card, replay = group_keys
        row = group.iloc[0]
        arena_str = str(row['arena'])
        arena = arena_str.replace('arena_','') if 'arena_' in arena_str else arena_str
        game_id = str(replay)
        x, y = find_placement_in_window(detections, arena, game_id, frame, window_size)
        mask = (
            (train_df['frame'] == frame) &
            (train_df['replay'] == replay) &
            (train_df['card'] == card)
        )
        train_df.loc[mask, 'x'] = x
        train_df.loc[mask, 'y'] = y

        if x != -1 and y != 1:
            matched_count += mask.sum()
        else:
            unmatched_count += mask.sum()
    """
    for idx, row in train_df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(train_df)} rows ({matched_count} matched, {unmatched_count} unmatched)")
        arena_str = str(row['arena'])
        arena = arena_str.replace('arena_', '') if 'arena_' in arena_str else arena_str
        game_id = str(row['replay'])
        frame = int(row['frame'])
        x, y = find_placement_in_window(detections, arena, game_id, frame, window_size)
        
        train_df.at[idx, 'x'] = x
        train_df.at[idx, 'y'] = y
        if x != -1 and y != 1:
            matched_count += 1
        else:
            unmatched_count += 1
    """
    print(f"\nFinal: {matched_count} matched, {unmatched_count} unmatched")
    print("\nSaving updated parquet")
    original_df = pd.read_parquet(parquet_path)
    original_df['x'] = train_df['x']
    original_df['y'] = train_df['y']
    original_df.to_parquet(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"Total rows: {len(train_df)}")
    print(f"Matched placements: {matched_count} ({matched_count/len(train_df)*100:.1f}%)")
    print(f"Unmatched placements: {unmatched_count} ({unmatched_count/len(train_df)*100:.1f}%)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", 
                       help="Path to training parquet file")
    parser.add_argument("detection_csv",
                       help="Path to detection CSV from create_placement_dataset.py")
    parser.add_argument("--output", "-o", required=True,
                       help="Output parquet file")
    args = parser.parse_args()

    if not Path(args.parquet_path).exists():
        print(f"Error: parquet file not found: {args.parquet_path}")
        return
    
    if not Path(args.detection_csv).exists():
        print(f"Error: detection csv not found: {args.detection_csv}")
        return

    update_training_parquet(
        args.parquet_path,
        args.detection_csv,
        args.output
    )


if __name__ == "__main__":
    main()
