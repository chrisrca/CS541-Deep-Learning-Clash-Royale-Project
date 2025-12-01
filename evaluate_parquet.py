import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


def evaluate_parquet(parquet_path: str):
    """
    Evaluate detections in parquet file.
    """
    print("PARQUET DETECTION EVALUATION")
    print("="*70)
    columns = ['card', 'x', 'y', 'arena', 'replay', 'frame']
    df = pd.read_parquet(parquet_path, columns=columns)
    
    print(f"Total rows: {len(df)}")
    # Calculate detection stats
    total = len(df)
    detected = len(df[(df['x'] != -1) & (df['y'] != -1)])
    missing = total - detected
    detection_rate = (detected / total * 100) if total > 0 else 0
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total placements:     {total:>8,}")
    print(f"Detected:             {detected:>8,}  ({detection_rate:.1f}%)")
    print(f"Missing (x=-1, y=-1): {missing:>8,}  ({100-detection_rate:.1f}%)")
    
    # Breakdown by arena
    print("\n" + "="*70)
    print("BREAKDOWN BY ARENA")
    print("="*70)
    print(f"{'Arena':<15} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
    print("-"*70)
    
    for arena in sorted(df['arena'].unique()):
        arena_df = df[df['arena'] == arena]
        arena_total = len(arena_df)
        arena_detected = len(arena_df[(arena_df['x'] != -1) & (arena_df['y'] != -1)])
        arena_missing = arena_total - arena_detected
        arena_rate = (arena_detected / arena_total * 100) if arena_total > 0 else 0
        
        print(f"{arena:<15} {arena_total:>8,} {arena_detected:>10,} {arena_missing:>10,} {arena_rate:>7.1f}%")
    
    # Breakdown by replay
    print("\n" + "="*70)
    print("BREAKDOWN BY REPLAY (Top 20 by count)")
    print("="*70)
    print(f"{'Replay ID':<40} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
    print("-"*70)
    
    replay_stats = []
    for replay in df['replay'].unique():
        replay_df = df[df['replay'] == replay]
        replay_total = len(replay_df)
        replay_detected = len(replay_df[(replay_df['x'] != -1) & (replay_df['y'] != -1)])
        replay_missing = replay_total - replay_detected
        replay_rate = (replay_detected / replay_total * 100) if replay_total > 0 else 0
        replay_stats.append((replay, replay_total, replay_detected, replay_missing, replay_rate))
    
    # Sort by total count and show top 20
    replay_stats.sort(key=lambda x: x[1], reverse=True)
    for replay, total, detected, missing, rate in replay_stats[:20]:
        replay_short = replay[:37] + "..." if len(replay) > 40 else replay
        print(f"{replay_short:<40} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")
    
    if len(replay_stats) > 20:
        print(f"... and {len(replay_stats) - 20} more replays")
    
    # Breakdown by card type 
    print("\n" + "="*70)
    print("BREAKDOWN BY CARD TYPE")
    print("="*70)
    print(f"{'Card':<20} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
    print("-"*70)
        
    card_stats = []
    for card in sorted(df['card'].unique()):
        card_df = df[df['card'] == card]
        card_total = len(card_df)
        card_detected = len(card_df[(card_df['x'] != -1) & (card_df['y'] != -1)])
        card_missing = card_total - card_detected
        card_rate = (card_detected / card_total * 100) if card_total > 0 else 0
        card_stats.append((card, card_total, card_detected, card_missing, card_rate))
        
    # Sort by detection rate (ascending - worst first)
    card_stats.sort(key=lambda x: x[4])
    for card, total, detected, missing, rate in card_stats:
        print(f"{card:<20} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")

    # Cards with highest number of misses
    print("\n" + "="*70)
    print("CARDS WITH HIGHEST NUMBER OF MISSES")
    print("="*70)
    print(f"{'Card':<20} {'Misses':>10} {'Total':>8} {'Miss Rate':>10}")
    print("-"*55)
    # Sort by number of misses descending
    card_stats.sort(key=lambda x: x[3], reverse=True)
    for card, total, detected, missing, rate in card_stats[:10]:
        miss_rate = (missing / total * 100) if total > 0 else 0
        print(f"{card:<20} {missing:>10,} {total:>8,} {miss_rate:>9.1f}%")

    # Frame range stats
    if detected > 0:
        detected_df = df[(df['x'] != -1) & (df['y'] != -1)]
        print("\n" + "="*70)
        print("COORDINATE STATISTICS (Detected placements only)")
        print("="*70)
        print(f"X coordinate range: {detected_df['x'].min():.0f} - {detected_df['x'].max():.0f}")
        print(f"Y coordinate range: {detected_df['y'].min():.0f} - {detected_df['y'].max():.0f}")
        print(f"Frame range:        {detected_df['frame'].min():.0f} - {detected_df['frame'].max():.0f}")
    
    # Low detection replays
    low_threshold = 30.0  # Less than 30% detection
    low_detection_replays = [r for r in replay_stats if r[4] < low_threshold]
    
    if low_detection_replays:
        print("\n" + "="*70)
        print(f"REPLAYS WITH LOW DETECTION (<{low_threshold}%)")
        print("="*70)
        print(f"{'Replay ID':<40} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
        print("-"*70)
        
        for replay, total, detected, missing, rate in sorted(low_detection_replays, key=lambda x: x[4]):
            replay_short = replay[:37] + "..." if len(replay) > 40 else replay
            print(f"{replay_short:<40} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path",
                       help="Path to parquet file with x,y coordinates")
    args = parser.parse_args()
    evaluate_parquet(
        args.parquet_path
    )
if __name__ == "__main__":
    main()
