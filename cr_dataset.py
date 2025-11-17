import cv2
import numpy as np
import os
import math
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download, login
from functools import partial
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from cr_element import CRElement

class CRDataset:
    def __init__(self):
        self.fs = HfFileSystem()
        self.url = "datasets/chrisrca/clash-royale-tv-replays/"
        self.repo_id = "chrisrca/clash-royale-tv-replays"
        self.arenas = self._load_arena_names()
        self.replay_names = self._load_replay_names()
        self.elixirs = [CRElement.Elixir(path) for path in os.listdir("elixir/")]
        self.cards = [CRElement.Card(path) for path in os.listdir("cards/")]

    def _load_elixirs(self) -> list[CRElement.Elixir]:
        return 

    def _load_arena_names(self) -> list[str]:
        arenas = self.fs.ls(self.url, detail=False, refresh=True)
        arenas = [item.split(self.url)[1] for item in arenas]
        arenas = [item for item in arenas if "arena" in item]
        return arenas

    def _load_replay_names(self) -> list[str]:
        replays = []
        for arena in self.arenas:
            replays.extend(self.fs.ls(self.url + arena, detail=False, refresh=True))
        replays = [item.split(self.url)[1] for item in replays]
        return replays

    def _get_table_from_db(self, replay) -> pq.ParquetDataset:
        with open('HUGGING_KEY.txt', 'r') as f:
            print(replay)
            path = hf_hub_download(repo_id=self.repo_id, filename=f"{replay}/frames.parquet", repo_type="dataset", token=f.read())
            print(path)
            return pq.read_table(path)
    
    def load_replay(self, replay_name = str) -> list[np.ndarray]:
        parquet = self._get_table_from_db(replay_name)
        column = parquet["image"]
        dataset = [item.as_py()['bytes'] for item in column]
        dataset = [cv2.cvtColor(np.array(Image.open(BytesIO(img))), cv2.COLOR_RGB2BGR) for img in dataset]
        return dataset
    
    def movement_highlight(backdrop: np.ndarray, replay: list[np.ndarray]):
        arena_start = CRElement.cut_to_fit(replay[0], CRElement.arena) # TODO have eacha arena have a template backdrop

        h0, w0 = arena_start.shape[:2]
        writer = cv2.VideoWriter("differences.mp4", 0, 10, (w0, h0), isColor=True)
        for image in replay:
            snipped = CRElement.cut_to_fit(image, CRElement.arena)
            diff = cv2.absdiff(snipped, arena_start)  # per-pixel absolute difference
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # collapse to grayscale if color
            _, mask = cv2.threshold(mask, 55, 255, cv2.THRESH_BINARY)  # highlight significant differences
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

            # Shrink (erode) then grow (dilate)
            mask_clean = cv2.erode(mask, kernel_erode, iterations=1)
            mask_clean = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

            blurred = cv2.GaussianBlur(mask_clean, (15,15), 0)
            _, mask_clean = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            # Option 2: show only changed regions
            changed_regions = cv2.bitwise_and(snipped, snipped, mask=mask_clean)
            writer.write(changed_regions)

        writer.release()

    

    
