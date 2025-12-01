import argparse
from pathlib import Path
from typing import Optional
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
from PIL import Image
from io import BytesIO

def open_image_cell(cell):
    # cell may be dict {"bytes": ...} or raw bytes
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray)):
        raw = cell
    else:
        raise ValueError(f"Unsupported image cell type: {type(cell)}")
    return Image.open(BytesIO(raw)).convert("RGB")

def extract_one_parquet(
    parquet_path: Path,
    image_col: str = "image",
    out_subdir: str = "images",
    limit: Optional[int] = None,
    overwrite: bool = False,
    batch_size: int = 512,
    ext: str = "png",
    prefix: str = "frame_",
) -> int:
    parent = parquet_path.parent
    out_dir = parent / out_subdir
    out_dir.mkdir(exist_ok=True)

    existing = list(out_dir.glob(f"{prefix}*.{ext}"))
    if existing and not overwrite:
        print(f"[SKIP] {parquet_path} -> images already exist in {out_dir} (use --overwrite to re-extract)")
        return 0
    if overwrite and existing:
        for p in existing:
            p.unlink()

    pf = ParquetFile(str(parquet_path))
    schema = pf.schema_arrow
    if image_col not in schema.names:
        print(f"[WARN] No '{image_col}' column in {parquet_path}, available: {schema.names}")
        return 0

    total_rows = pf.metadata.num_rows
    to_process = min(limit, total_rows) if limit else total_rows

    written = 0
    row_base = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[image_col]):
        if limit is not None and row_base >= to_process:
            break
        col = batch[image_col]
        rows_left = to_process - row_base if limit is not None else len(col)
        rows_this_batch = min(len(col), rows_left)

        for i in range(rows_this_batch):
            cell = col[i].as_py()
            try:
                img = open_image_cell(cell)
            except Exception as e:
                print(f"[WARN] {parquet_path} row {row_base + i} failed: {e}")
                continue
            out_path = out_dir / f"{prefix}{row_base + i:06d}.{ext}"
            img.save(out_path)
            written += 1

        row_base += len(col)
        if limit is not None and row_base >= to_process:
            break

    print(f"[DONE] {parquet_path} -> wrote {written} images in {out_dir}")
    return written

def find_parquets(root: Path, parquet_name: str, arena_filter: Optional[str] = None, game_filter: Optional[str] = None):
    """
    Only look one level down for directories named arena_*,
    then search inside those (recursively) for parquet_name.
    
    Args:
        root: Root directory containing arena_* folders
        parquet_name: Name of parquet file to find
        arena_filter: If specified, only process this specific arena (e.g., "arena_11" or "11")
        game_filter: If specified, only process games matching this pattern (e.g., "game_01" or UUID)
    """
    parquets = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("arena_"):
            continue
        
        # Check arena filter
        if arena_filter:
            arena_match = arena_filter if arena_filter.startswith("arena_") else f"arena_{arena_filter}"
            if child.name != arena_match:
                continue
        
        # If game filter specified, only look in matching game directories
        if game_filter:
            for game_dir in child.iterdir():
                if game_dir.is_dir() and game_filter in game_dir.name:
                    game_parquets = list(game_dir.rglob(parquet_name))
                    parquets.extend(game_parquets)
        else:
            # No game filter, get all parquets in this arena
            parquets.extend(child.rglob(parquet_name))
    
    return sorted(parquets)

def main():
    ap = argparse.ArgumentParser(description="Extract images adjacent to Parquet files under arena_* folders.")
    ap.add_argument("--root", default="E:\Projects\CS541\ClashRoyale\hf_subset",
                    help="Root containing arena_* folders.")
    ap.add_argument("--parquet-name", default="frames.parquet",
                    help="Parquet filename to find inside arena subfolders.")
    ap.add_argument("--arena", type=str, default=None,
                    help="Extract only from specific arena (e.g., 'arena_11' or '11').")
    ap.add_argument("--game", type=str, default=None,
                    help="Extract only from games matching this pattern (e.g., 'game_01' or UUID substring).")
    ap.add_argument("--image-col", default="image",
                    help="Column name holding image bytes.")
    ap.add_argument("--out-subdir", default="images",
                    help="Images output subfolder created next to each parquet.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit frames per parquet (default all).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if images exist.")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Arrow batch size.")
    ap.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"],
                    help="Image extension.")
    ap.add_argument("--prefix", default="frame_",
                    help="Filename prefix for frames.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List parquets found but do not extract.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root does not exist: {root}")
        return

    filter_msg = []
    if args.arena:
        filter_msg.append(f"arena={args.arena}")
    if args.game:
        filter_msg.append(f"game={args.game}")
    filter_str = f" (filters: {', '.join(filter_msg)})" if filter_msg else ""

    parquets = find_parquets(root, args.parquet_name, arena_filter=args.arena, game_filter=args.game)
    if not parquets:
        print(f"[INFO] No '{args.parquet_name}' files found under arenas in {root}{filter_str}")
        return

    print(f"[INFO] Found {len(parquets)} parquet files{filter_str}:")
    for p in parquets:
        print(f"  - {p}")

    if args.dry_run:
        print("[INFO] Dry run complete — no extraction performed.")
        return

    grand_total = 0
    for pq_path in parquets:
        grand_total += extract_one_parquet(
            pq_path,
            image_col=args.image_col,
            out_subdir=args.out_subdir,
            limit=args.limit,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
            ext=args.ext,
            prefix=args.prefix,
        )

    print(f"[TOTAL] Extracted {grand_total} images across {len(parquets)} parquet files.")

if __name__ == "__main__":
    main()