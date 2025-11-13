import webdataset as wds
import io, json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

# 1) list of shard URLs (on Hugging Face raw files or an HTTP/S endpoint)
# e.g. ["https://.../shard-0000.tar", "https://.../shard-0001.tar", ...]
shard_urls = ["https://your-hf-repo/.../shard-0000.tar", "https://your-hf-repo/.../shard-0001.tar"]

# 2) create WebDataset pipeline
dataset = (
    wds.WebDataset(shard_urls, resampled=True)   # resampled=True supports epoch-level randomness
      .decode()                                  # decode from bytes -> python types for simple cases
      .to_tuple("jpg", "extra.json", "label.json")  # extract those file types for each sample
)

# 3) optional mapping to combine frames list (if you saved one jpg per frame name pattern)
def transform(sample):
    jpg_bytes, extra_json, label_json = sample
    # decode frames: if jpg_bytes is a list of frames you'll get them; else adapt
    # For simplicity assume frames are provided as a single concatenated numpy npz or a single jpg with stacked frames.
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB").resize((160,160))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))
    frames = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W) or stack more
    extra = torch.tensor(json.loads(extra_json.decode("utf-8"))["extra_feats"], dtype=torch.float32)
    label_obj = json.loads(label_json.decode("utf-8"))
    if label_obj is None:
        label_card = -1
        label_xy = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    else:
        label_card = int(label_obj["card"])
        label_xy = torch.tensor([label_obj["x"], label_obj["y"]], dtype=torch.float32)
    return {"frames": frames, "extra": extra, "label_card": label_card, "label_xy": label_xy}

dataset = dataset.map(transform)

# 4) DataLoader
loader = DataLoader(dataset, batch_size=16, num_workers=4)
for batch in loader:
    frames = batch["frames"]    # (B, T, C, H, W) if you stacked T frames
    extra = batch["extra"]
    labels = batch["label_card"]
    # ...
