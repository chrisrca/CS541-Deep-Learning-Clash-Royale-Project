import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import random
data_path = "training.parquet"
image_root = "/home/ostikar/MyProjects/CS541/ClashRoyale/hf_subset"
df = pd.read_parquet(data_path)
#drop rows with missing x/y/replay
samples = df.dropna(subset=['x', 'y', 'replay'])
print(f"Rows with x, y, replay: {len(samples)}")

#select 100 samples
samples_100 = samples.sample(n=min(100, len(samples)), random_state=42)
output_dir = "placement_overlay_samples"
os.makedirs(output_dir, exist_ok=True)

for idx, row in samples_100.iterrows():
    replay = row['replay']
    arena = str(row['arena'])
    x, y = int(row['x']), int(row['y'])
    frame = int(row['frame']) if 'frame' in row and not pd.isnull(row['frame']) else None
    print(f"Processing replay: {replay}, frame: {frame}, x: {x}, y: {y}")
    img_dir = os.path.join(image_root, arena, str(replay), "images")
    if not os.path.isdir(img_dir):
        print(f"Image directory does not exist: {img_dir}")
        continue
    if frame is None:
        print(f"Frame is None for replay {replay}")
        continue
    img_filename = f"frame_{frame:06d}.png"
    img_path = os.path.join(img_dir, img_filename)
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = 8
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0), outline=(0,0,0))
        overlay_dir = os.path.join(output_dir, arena, str(replay))
        os.makedirs(overlay_dir, exist_ok=True)
        save_path = os.path.join(overlay_dir, f"frame_{frame:06d}_{x}_{y}.png")
        img.save(save_path)
        print(f"Saved overlay: {save_path}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")