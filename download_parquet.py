from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "chrisrca/clash-royale-tv-replays"
filename = "training_offset_1.parquet"

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)

# Copy to current directory
target_path = os.path.join(os.getcwd(), filename)
shutil.copy(local_path, target_path)
print(f"Downloaded to: {target_path}")