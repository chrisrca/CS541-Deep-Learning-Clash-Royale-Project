import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from huggingface_hub import list_repo_files, hf_hub_download

DATASET_ID = "chrisrca/clash-royale-tv-replays"

# by_arena: {"arena_02": [(game_uuid, "arena_02/<uuid>/frames.parquet"), ...], ...}
def discover_frames(dataset_id: str) -> Dict[str, List[Tuple[str, str]]]:
    files = list_repo_files(dataset_id, repo_type="dataset")
    frame_paths = [f for f in files if f.endswith("frames.parquet")]
    by_arena = {}
    for p in frame_paths:
        parts = p.split("/")
        # Expect: arena_xx/<uuid>/frames.parquet
        if len(parts) >= 3 and parts[0].startswith("arena_"):
            arena = parts[0]
            game_folder = parts[1]
            by_arena.setdefault(arena, []).append((game_folder, p))
    return by_arena

def sample_frames(by_arena: Dict[str, List[Tuple[str, str]]],
                  arenas: List[str],
                  per_arena: int,
                  seed: int) -> List[Tuple[str, str, str]]:
    random.seed(seed)
    selected = []
    for arena in arenas:
        candidates = by_arena.get(arena, [])
        if not candidates:
            print(f"[WARN] No frames.parquet entries found for {arena}")
            continue
        if per_arena >= len(candidates):
            chosen = candidates
        else:
            chosen = random.sample(candidates, per_arena)
        selected.extend([(arena, game_folder, repo_path) for (game_folder, repo_path) in chosen])
        print(f"[INFO] Selected {len(chosen):>3} from {arena} (available: {len(candidates)})")
    return selected

def find_specific_game(by_arena: Dict[str, List[Tuple[str, str]]], 
                       game_id: str) -> Optional[Tuple[str, str, str]]:
    for arena, games in by_arena.items():
        for game_folder, repo_path in games:
            if game_folder == game_id:
                return (arena, game_folder, repo_path)
    return None

def download_selected(dataset_id: str,
                      selection: List[Tuple[str, str, str]],
                      out_dir: Path,
                      overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for arena, game_folder, repo_path in selection:
        local_game_dir = out_dir / arena / game_folder
        local_game_dir.mkdir(parents=True, exist_ok=True)
        local_target = local_game_dir / "frames.parquet"
        if local_target.exists() and not overwrite:
            print(f"[SKIP] Exists: {local_target}")
            continue
        print(f"[DL] {repo_path} -> {local_target}")
        cache_path = hf_hub_download(
            repo_id=dataset_id,
            filename=repo_path,
            repo_type="dataset"
        )
        local_target.write_bytes(Path(cache_path).read_bytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--game", type=str, help="Download a specific game UUID (e.g., from arena_31/<uuid>). Overrides sampling.")
    ap.add_argument("--arenas", nargs="+", help="Arena folders (e.g. arena_01 arena_02). If absent, use all discovered.")
    ap.add_argument("--all-arenas", action="store_true", help="Use all discovered arenas (overrides --arenas)")
    ap.add_argument("--per-arena", type=int, default=3, help="Minimum/target frames.parquet per arena")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../hf_subset", help="Output directory (default: ../hf_subset)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    by_arena = discover_frames(args.dataset)
    discovered = sorted(by_arena.keys())
    print(f"[INFO] Discovered arenas: {discovered}")
    if args.game:
        result = find_specific_game(by_arena, args.game)
        if result is None:
            print(f"[ERROR] Game '{args.game}' not found in any arena.")
            return
        arena, game_folder, repo_path = result
        print(f"[INFO] Found game '{args.game}' in {arena}")
        selection = [(arena, game_folder, repo_path)]
    else:
        if args.all_arenas or not args.arenas:
            arenas = discovered
        else:
            arenas = args.arenas

        if not arenas:
            print("error: no arenas selected or discovered."); return

        selection = sample_frames(by_arena, arenas, args.per_arena, args.seed)
        print(f"Total files to download: {len(selection)}")

    download_selected(args.dataset, selection, Path(args.out), overwrite=args.overwrite)
    print("[DONE]")

if __name__ == "__main__":
    main()