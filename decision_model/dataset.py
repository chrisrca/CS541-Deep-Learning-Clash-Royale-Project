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
    "evo_archers", "evo_baby_dragon", "evo_barbs", "evo_bats", "evo_battle_ram", "evo_bomber", "evo_cannon", "evo_dart_goblin",
    "evo_electro_dragon", "evo_executioner", "evo_firecracker", "evo_ghost", "evo_goblin_barrel", "evo_goblin_cage", 
    "evo_goblin_drill", "evo_goblin_giant", "evo_ice_spirit", "evo_inferno_dragon", "evo_knight", "evo_lumberjack", "evo_mortar", 
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
    "skeleton_king", "skeletons", "snowball", "sparky", "spear_goblins", "spirit_empress", "spirit_empress_dragon", "tesla_coil", "tombstone", "tornado", 
    "valk", "vines", "void", "wallbreakers", "witch", "wizard", "xbow", "zap", "zappies"
]

# Sort to ensure consistent ID mapping
ALL_CARDS.sort()
CARD_TO_ID = {name: i for i, name in enumerate(ALL_CARDS)}

# Grid discretization parameters (pixel to tile conversion)
# These define the playable area within the image
IMAGE_WIDTH = 432
IMAGE_HEIGHT = 680
Y_OFFSET_TOP = 62
Y_OFFSET_BOTTOM = 7
X_OFFSET_LEFT = 0
X_OFFSET_RIGHT = 0
NUM_COLS = 18
NUM_ROWS = 32

# Calculate tile dimensions from offsets
GRID_WIDTH = IMAGE_WIDTH - X_OFFSET_LEFT - X_OFFSET_RIGHT
GRID_HEIGHT = IMAGE_HEIGHT - Y_OFFSET_TOP - Y_OFFSET_BOTTOM
TILE_WIDTH = GRID_WIDTH / NUM_COLS
TILE_HEIGHT = GRID_HEIGHT / NUM_ROWS

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

        # Filter out samples with invalid hand data
        print("Filtering out samples with invalid data...")
        self.valid_indices = []

        for i in range(self.table.num_rows):
            row_table = self.table.slice(i, 1)
            data = row_table.to_pydict()
            card_played = data["card"][0]

            # Validate hand data if present
            if "hand" in data:
                # Check if hand column has 4 entries
                cards_in_hand = data["hand"][0]
                if len(cards_in_hand) != 4:
                    continue

                # Check if card played is in hand
                if card_played != "None" and card_played not in cards_in_hand:
                    continue

            # Exclude samples where card was played but placement is unknown (-1, -1)
            # This indicates low confidence in placement detection
            # Note: x and y columns may be missing entirely when card is "None"
            if card_played != "None":
                x_val = int(data["x"][0]) if "x" in data else -1
                y_val = int(data["y"][0]) if "y" in data else -1
                if x_val == -1 and y_val == -1:
                    continue

            # Keep sample if it has a valid hand
            self.valid_indices.append(i)

        print(f"Kept {len(self.valid_indices)} valid samples out of {self.table.num_rows} total samples.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Map to actual table index
        table_idx = self.valid_indices[idx]

        # Slicing the in-memory table is efficient
        row_table = self.table.slice(table_idx, 1)
        data = row_table.to_pydict()

        # Decode the image from bytes
        raw_image_bytes = data["png_bytes"][0]
        
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
        
        if card_name in CARD_TO_ID:
            card_id = CARD_TO_ID[card_name]
        elif card_name == "None":
            card_id = len(CARD_TO_ID)
        else:
            raise ValueError(f"Unknown card name: {card_name}")

        # Handle case where x/y columns may be missing when card is "None"
        if card_name == "None":
            tile_index = -1
        else:
            pixel_x = int(data["x"][0])
            pixel_y = int(data["y"][0])
            
            # Convert pixel coordinates to tile coordinates
            tile_x = int((pixel_x - X_OFFSET_LEFT) / TILE_WIDTH)
            tile_y = int((pixel_y - Y_OFFSET_TOP) / TILE_HEIGHT)
            
            # Clamp to valid range
            tile_x = max(0, min(tile_x, NUM_COLS - 1))
            tile_y = max(0, min(tile_y, NUM_ROWS - 1))

            tile_index = tile_y * NUM_COLS + tile_x
        
        # Check if hand column is missing
        if "hand" not in data:
            # Mark all cards as in hand and all cards as playable
            playable_mask = torch.ones(self.num_cards + 1, dtype=torch.float32)
            hand_mask = torch.ones(self.num_cards + 1, dtype=torch.float32)
            print("WARNING: Hand data is missing from the dataset. Applying no masking.")
        else:
            hand_mask_ids = []
            playable_mask_ids = []
            
            for c in data["hand"][0]:
                # None value indicates empty slot in hand
                # Occurs during transitions after playing a card
                # but before new card is drawn, or at the start of the game
                if c is None:
                    continue

                if c in CARD_TO_ID:
                    playable_mask_ids.append(CARD_TO_ID[c])

                # if card is not playable (too expensive), it will be prefixed with "gray_"
                # in this case, it will not be included in the playable mask
                # however, we should still include it in the hand mask

                # first, we need to strip the "gray_" prefix if it exists
                if c.startswith("gray_"):
                    c = c[5:]

                if c in CARD_TO_ID:
                    hand_mask_ids.append(CARD_TO_ID[c])
                else:
                    raise ValueError(f"Unknown card name: {c}")

            playable_mask_ids = torch.tensor(playable_mask_ids, dtype=torch.long)
            # Allocate an extra slot for the No-Op action at index self.num_cards.
            playable_mask = torch.zeros(self.num_cards + 1, dtype=torch.float32)
            playable_mask[playable_mask_ids] = 1.0
            # No-Op is always legal
            playable_mask[-1] = 1.0

            hand_mask_ids = torch.tensor(hand_mask_ids, dtype=torch.long)
            hand_mask = torch.zeros(self.num_cards + 1, dtype=torch.float32)
            hand_mask[hand_mask_ids] = 1.0
        
        elixir = float(data["elixir"][0]) if "elixir" in data else 10.0
        blue_left_princess_tower_health = int(data["blue_left_princess_tower_health"][0]) if "blue_left_princess_tower_health" in data else 3000
        blue_right_princess_tower_health = int(data["blue_right_princess_tower_health"][0]) if "blue_right_princess_tower_health" in data else 3000
        blue_king_tower_health = int(data["blue_king_tower_health"][0]) if "blue_king_tower_health" in data else 5000
        red_left_princess_tower_health = int(data["red_left_princess_tower_health"][0]) if "red_left_princess_tower_health" in data else 3000
        red_right_princess_tower_health = int(data["red_right_princess_tower_health"][0]) if "red_right_princess_tower_health" in data else 3000
        red_king_tower_health = int(data["red_king_tower_health"][0]) if "red_king_tower_health" in data else 5000

        numeric_features = torch.tensor([elixir, blue_left_princess_tower_health, blue_right_princess_tower_health, blue_king_tower_health, red_left_princess_tower_health, red_right_princess_tower_health, red_king_tower_health], dtype=torch.float32)
        label_card = torch.tensor(card_id, dtype=torch.int32)
        label_placement = torch.tensor(tile_index, dtype=torch.long)

        return {
            "frames": frames,
            "playable_mask": playable_mask,
            "hand_mask": hand_mask,
            "numeric_features": numeric_features,
            "label_card": label_card,
            "label_placement": label_placement,
        }