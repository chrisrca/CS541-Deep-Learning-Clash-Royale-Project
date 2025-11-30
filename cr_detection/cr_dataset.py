import cv2
import numpy as np
import os
import json
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download, login
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from cr_detection.cr_element import CRElement
from cr_detection.cr_gamestate import CRGameState
import shutil
import uuid
import matplotlib.pyplot as plt


class CRDataset:
    def __init__(self):
        self.fs = HfFileSystem()
        self.url = "datasets/chrisrca/clash-royale-tv-replays/"
        self.repo_id = "chrisrca/clash-royale-tv-replays"
        self.arenas = self._load_arena_names()
        self.replay_names = self._load_replay_names()
        self._elixirs = [CRElement.Elixir(path) for path in os.listdir("cr_detection/elixir/")]
        self.elixirs = {elixir.value: elixir for elixir in self._elixirs}
        self._cards = [CRElement.Card(path) for path in os.listdir("cr_detection/cards/")]
        self.cards = {card.name: card for card in self._cards}
        with open("manifest.json", "r") as json_file:
            self.manifest = json.load(json_file)

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
            path = hf_hub_download(repo_id=self.repo_id, filename=f"{replay}/frames.parquet", repo_type="dataset", token=f.read())
            return pq.read_table(path)
    
    def load_replay(self, replay_name: str, clear_huggingface = False) -> tuple[list[np.ndarray], list[CRElement.Card], bool]:
        """

        """
        parquet = self._get_table_from_db(replay_name)
        column = parquet["image"]
        dataset = [item.as_py()['bytes'] for item in column]
        dataset = [cv2.cvtColor(np.array(Image.open(BytesIO(img))), cv2.COLOR_RGB2BGR) for img in dataset]
        deck, all_identified = self._find_cards_in_video(dataset, cull_selected_cards=False) # Set to true to scrape for new card images
        replay_metadata = self.get_replay_from_manifest(replay_name)
        if replay_metadata == None:
            self.manifest.append({'replay': replay_name, 'cards_identified': True})
            if(clear_huggingface):
                shutil.rmtree("C:\\Users\\0dps1\\.cache\\huggingface\\hub\\datasets--chrisrca--clash-royale-tv-replays\\blobs")
                shutil.rmtree("C:\\Users\\0dps1\\.cache\\huggingface\\hub\\datasets--chrisrca--clash-royale-tv-replays\\snapshots")
            with open("manifest.json", "w") as json_file:
                json.dump(self.manifest, json_file, indent=4)
        return dataset, deck, all_identified
    
    def get_replay_from_manifest(self, replay_name):
        return next((metadata for metadata in self.manifest if metadata['replay'] == replay_name), None)
    
    def _find_cards_in_video(self, replay, cull_selected_cards = True) -> tuple[list[CRElement.Card], bool]:
        """
        Return cards found and if any of them are unknown
        """

        seen_images = {}

        for i, frame in enumerate(replay):
            image = frame
            for card in CRElement.get_images_in_hand(image):
                CRDataset._check_exists_add_otherwise(seen_images, card, i, template_threshold=0.96, cull_selected_cards=cull_selected_cards)

            if i % 30 == 0 and i > 15 :
                seen_images = {key: value for key, value in seen_images.items() if CRDataset._is_persistent(value, i)}
        seen_images = {key: value for key, value in seen_images.items() if CRDataset._is_persistent(value, i, True)}

        deck = {}
        unknown_flag = False

        for name, obj in seen_images.items():
            image = obj[1]
            labelled_image_collection = {name: CRElement.prep(card.BGR) for name, card in self.cards.items()}
            best_score, best_match = CRElement.best_match(CRElement.prep(image), labelled_image_collection, shearing=(9,3))
            if(best_score < 0.80):
                print("Potential new card:", name)
                os.makedirs("potential_new_cards/", exist_ok=True)
                cv2.imwrite(f"potential_new_cards/{name}.png", image)
                unknown_flag = True
            else:
                deck[best_match] = self.cards[best_match]

        return deck, not unknown_flag
    
    @staticmethod
    def _check_exists_add_otherwise(library, newimage, frame, template_threshold, shearing = (3,3), cull_selected_cards = True):   
        if CRElement.is_grayscale(newimage) or (CRGameState.is_card_selected(newimage) and cull_selected_cards) or CRGameState.is_card_sliding(newimage):
            return

        # object: prepped image, original image, view count, timestamp, new_flag
        # library: hash: (object)
        if(len(library) == 0):
            library[str(uuid.uuid4())] = [CRElement.prep(newimage, 4), newimage, 1, frame]
            return
        labelled_image_collection = {key: obj[0] for key, obj in library.items()}
        best_score, best_match = CRElement.best_match(CRElement.prep(newimage, 4), labelled_image_collection, shearing=shearing)
        
        if best_score > template_threshold:
            library[best_match][2] = library[best_match][2] + 1
        else:
            library[str(uuid.uuid4())] = [CRElement.prep(newimage, 4), newimage, 1, frame]

    @staticmethod
    def _is_persistent(obj, time, override_window = False):
        count = obj[2]
        
        if(override_window):
            return count > 20
        
        age = time - obj[3]

        in_window = 15 < age < 45
        keep = count > 20 or not in_window
        return keep
    
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

    
