import argparse
from pathlib import Path
from datasets import load_dataset


def download_train_parquet(dataset_name: str, output_path: str = "train.parquet"):
    print(f"\nDownloading dataset: {dataset_name}")
    
    try:
        ds = load_dataset(dataset_name)
        
        print(f"dataset splits: {list(ds.keys())}")
        
        train_ds = ds['train']
        print(f"training rows: {len(train_ds)}")
        print(f"columns: {train_ds.column_names}")
        print(f"\nsaving to {output_path}")
        train_ds.to_parquet(output_path)
        
        file_size = Path(output_path).stat().st_size / (1024*1024)
        print(f"file: {output_path}")
        print(f"size: {file_size:.2f} MB")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="chrisrca/clash-royale-tv-replays", 
                       help="HuggingFace dataset name")
    parser.add_argument("--output", "-o", default="train.parquet",
                       help="Output parquet file")
    args = parser.parse_args()
    download_train_parquet(args.dataset, args.output)


if __name__ == "__main__":
    main()
