import io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

# List of known cards derived from file names
ALL_CARDS = [
    "archer_queen", "archers", "arrows", "baby_dragon", "balloon", "bandit", "barb_barrel", "barb_hut", "barbs", 
    "bats", "battle_ram", "berserker", "bomb_tower", "bomber", "boss_bandit", "bowler", "bush_goblin", 
    "caged_goblin", "cannon", "cannon_cart", "clone", "dark_prince", "dart_goblin", "e_barbs", "e_wiz", 
    "earthquake", "electro_dragon", "electro_giant", "electro_spirit", "elixir_golem", "elixir_pump", "empty", 
    "evo_archers", "evo_baby_dragon", "evo_barbs", "evo_bats", "evo_battle_ram", "evo_bomber", "evo_cannon", 
    "evo_electro_dragon", "evo_executioner", "evo_firecracker", "evo_ghost", "evo_goblin_barrel", "evo_goblin_cage", 
    "evo_goblin_drill", "evo_goblin_giant", "evo_ice_spirit", "evo_inferno_dragon", "evo_lumberjack", "evo_mortar", 
    "evo_musketeer", "evo_royal_giant", "evo_royal_hogs", "evo_royal_recruit", "evo_skele_barrel", 
    "evo_skeletion_army", "evo_skeletons", "evo_snowball", "evo_tesla_coil", "evo_valk", "evo_wallbreakers", 
    "evo_witch", "evo_wizard", "evo_zap", "executioner", "fire_spirit", "fireball", "firecracker", "fisherman", 
    "flying_machine", "freeze", "furnace", "ghost", "giant", "giant_skeleton", "goblin_barrel", "goblin_curse", 
    "goblin_demolisher", "goblin_drill", "goblin_gang", "goblin_giant", "goblin_hut", "goblin_machine", "goblins", 
    "goblinstein", "golden_knight", "golem", "graveyard", "guards", "heal_spirit", "healer", "hog_rider", "hunter", 
    "ice_golem", "ice_spirit", "ice_wizard", "inferno_dragon", "inferno_tower", "knight", "lava_hound", "lightning", 
    "little_prince", "log", "lumberjack", "magic_archer", "mega_knight", "mega_minion", "mega_miner", "mighty_miner", 
    "miner", "mini_pekka", "minion_horde", "minions", "monk", "mortar", "mother_witch", "musketeer", "musketeers", 
    "night_witch", "pekka", "phoenix", "poison", "prince", "princess", "rage", "ram_rider", "rascals", "rocket", 
    "royal_delivery", "royal_giant", "royal_hogs", "royal_recruits", "skarmy", "skele_barrel", "skeleton_dragons", 
    "skeleton_king", "skeletons", "snowball", "sparky", "spear_goblins", "tesla_coil", "tombstone", "tornado", 
    "valk", "vines", "wallbreakers", "witch", "xbow", "zap", "zappies"
]

# Sort to ensure consistent ID mapping
ALL_CARDS.sort()
CARD_TO_ID = {name: i for i, name in enumerate(ALL_CARDS)}

class ClashRoyaleDataset(Dataset):
    def __init__(self, files, grid_w, grid_h, num_cards):
        super().__init__()
        self.files = list(files)
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.num_cards = num_cards

        if not self.files:
            raise ValueError("No parquet files provided to ClashRoyaleDataset")

        print(f"Loading {len(self.files)} parquet files into memory...")
        tables = []
        for path in self.files:
            tables.append(pq.read_table(path))
        
        self.table = pa.concat_tables(tables)
        print(f"Loaded {self.table.num_rows} rows.")

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Slicing the in-memory table is efficient
        row_table = self.table.slice(idx, 1)
        data = row_table.to_pydict()

        # Decode the image from bytes
        raw_image_bytes = data["png_bytes"][0]
        
        # Check if raw_bytes is a list/array (multiple frames) or single bytes (one frame)
        # pyarrow might return a list if it's a list column, or bytes if it's binary.
        # However, the user said "stored in png_bytes", implying the column content.
        # If it is a list of binaries, `data["png_bytes"][0]` should be a list.
        # If it is a single binary, it is bytes.
        
        frame_list = []
        if isinstance(raw_image_bytes, (list, np.ndarray)):
            # Multiple frames
            for b in raw_image_bytes:
                img = Image.open(io.BytesIO(b))
                t = torch.from_numpy(np.array(img).astype(np.float32))
                # (H, W, C) -> (C, H, W)
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
                frame_list.append(t)
            frames = torch.stack(frame_list, dim=0) # (T, C, H, W)
        else:
            # Single frame
            img = Image.open(io.BytesIO(raw_image_bytes))
            t = torch.from_numpy(np.array(img).astype(np.float32))
            # (H, W, C) -> (C, H, W)
            if t.ndim == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            frames = t.unsqueeze(0) # (1, C, H, W)

        frames = frames / 255.0

        # Parse card name and convert to ID
        card_name = data["card"][0]
        if isinstance(card_name, bytes):
            card_name = card_name.decode('utf-8')
        
        if card_name in CARD_TO_ID:
            card_id = CARD_TO_ID[card_name]
        elif card_name == "none":
            card_id = len(CARD_TO_ID)
        else:
            raise ValueError(f"Unknown card name: {card_name}")

        tile_x = int(data["x"][0])
        tile_y = int(data["y"][0])
        
        if "cards_in_hand" in data:
            # cards_in_hand might be a list of strings/bytes
            raw_hand = data["cards_in_hand"][0]
            hand_ids = []
            for c in raw_hand:
                if isinstance(c, bytes):
                    c = c.decode('utf-8')
                if c in CARD_TO_ID:
                    hand_ids.append(CARD_TO_ID[c])
                else:
                    raise ValueError(f"Unknown card name in hand: {c}")

            cards_in_hand = torch.tensor(hand_ids, dtype=torch.long)
            # Allocate an extra slot for the No-Op action at index self.num_cards.
            mask = torch.zeros(self.num_cards + 1, dtype=torch.float32)
            mask[cards_in_hand] = 1.0
            # No-Op is always legal
            mask[-1] = 1.0
        else:
            # All cards plus No-Op are legal
            mask = torch.ones(self.num_cards + 1, dtype=torch.float32)
        
        elixir = float(data["elixir"][0]) if "elixir" in data else 10.0
        blue_left_princess_tower_health = int(data["blue_left_princess_tower_health"][0]) if "blue_left_princess_tower_health" in data else 3000
        blue_right_princess_tower_health = int(data["blue_right_princess_tower_health"][0]) if "blue_right_princess_tower_health" in data else 3000
        blue_king_tower_health = int(data["blue_king_tower_health"][0]) if "blue_king_tower_health" in data else 5000
        red_left_princess_tower_health = int(data["red_left_princess_tower_health"][0]) if "red_left_princess_tower_health" in data else 3000
        red_right_princess_tower_health = int(data["red_right_princess_tower_health"][0]) if "red_right_princess_tower_health" in data else 3000
        red_king_tower_health = int(data["red_king_tower_health"][0]) if "red_king_tower_health" in data else 5000

        numeric_features = torch.tensor([elixir, blue_left_princess_tower_health, blue_right_princess_tower_health, blue_king_tower_health, red_left_princess_tower_health, red_right_princess_tower_health, red_king_tower_health], dtype=torch.float32)

        label_card = torch.tensor(card_id, dtype=torch.int32) # equal to num_cards if no card was played
        
        # If tile_x is invalid OR card is "none", treat as No-Op
        # "none" in CARD_TO_ID has an ID. 
        # However, "none" implies No-Op.
        # If card_name == "none", we should map it to num_cards (No-Op class).
        
        if card_name == "none" or tile_x < 0 or tile_y < 0:
            tile_index = -1
        else:
            tile_index = tile_y * self.grid_w + tile_x

        label_card = torch.tensor(card_id, dtype=torch.int32)
        label_placement = torch.tensor(tile_index, dtype=torch.long)

        return {
            "frames": frames,
            "mask": mask,
            "numeric_features": numeric_features,
            "label_card": label_card,
            "label_placement": label_placement,
        }