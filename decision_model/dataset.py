import io
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# List of known cards derived from file names
ALL_CARDS = [
    "archer_queen", "archers", "arrows", "baby_dragon", "balloon", "bandit", "barb_barrel", "barb_hut", "barbs", 
    "bats", "battle_ram", "berserker", "bomb_tower", "bomber", "boss_bandit", "bowler", "bush_goblin", 
    "caged_goblin", "cannon", "cannon_cart", "clone", "dark_prince", "dart_goblin", "e_barbs", "e_wiz", 
    "earthquake", "electro_dragon", "electro_giant", "electro_spirit", "elixir_golem", "elixir_pump", 
    "evo_archers", "evo_baby_dragon", "evo_barbs", "evo_bats", "evo_battle_ram", "evo_bomber", "evo_cannon", "evo_dart_goblin",
    "evo_electro_dragon", "evo_executioner", "evo_firecracker", "evo_furnace", "evo_ghost", "evo_goblin_barrel", "evo_goblin_cage", 
    "evo_goblin_drill", "evo_goblin_giant", "evo_hunter", "evo_ice_spirit", "evo_inferno_dragon", "evo_knight", "evo_lumberjack", "evo_mega_knight", "evo_mortar", 
    "evo_musketeer", "evo_pekka", "evo_royal_giant", "evo_royal_hogs", "evo_royal_recruit", "evo_skele_barrel", 
    "evo_skeletion_army", "evo_skeletons", "evo_snowball", "evo_tesla_coil", "evo_valk", "evo_wallbreakers", 
    "evo_witch", "evo_wizard", "evo_zap", "executioner", "fire_spirit", "fireball", "firecracker", "fisherman", 
    "flying_machine", "freeze", "furnace", "ghost", "giant", "giant_skeleton", "goblin_barrel", "goblin_curse", 
    "goblin_demolisher", "goblin_drill", "goblin_gang", "goblin_giant", "goblin_hut", "goblin_machine", "goblins", 
    "goblinstein", "golden_knight", "golem", "graveyard", "guards", "heal_spirit", "healer", "hog_rider", "hunter", 
    "ice_golem", "ice_spirit", "ice_wizard", "inferno_dragon", "inferno_tower", "knight", "lava_hound", "lightning", 
    "little_prince", "log", "lumberjack", "magic_archer", "mega_knight", "mega_minion", "mega_miner", "mighty_miner", 
    "miner", "mini_pekka", "minion_horde", "minions", "monk", "mortar", "mother_witch", "musketeer", "musketeers", 
    "night_witch", "pekka", "phoenix", "poison", "prince", "princess", "rage", "ram_rider", "rascals", "rocket", 
    "royal_delivery", "royal_giant", "royal_hogs", "royal_recruits", "rune_giant", "skarmy", "skele_barrel", "skeleton_dragons", 
    "skeleton_king", "skeletons", "snowball", "sparky", "spear_goblins", "spirit_empress", "spirit_empress_dragon", "tesla_coil", "tombstone", "tornado", 
    "valk", "vines", "void", "wallbreakers", "witch", "wizard", "xbow", "zap", "zappies"
]

# Sort to ensure consistent ID mapping
ALL_CARDS.sort()
CARD_TO_ID = {name: i for i, name in enumerate(ALL_CARDS)}

# Spell cards to exclude from training/testing
SPELL_CARDS = {
    "arrows", "barb_barrel", "clone", "earthquake", "fireball", "freeze", 
    "goblin_barrel", "goblin_curse", "graveyard", "lightning", "log", "poison", 
    "rage", "rocket", "royal_delivery", "snowball", "tornado", "void", "zap",
    "evo_goblin_barrel", "evo_snowball", "evo_zap"
}

# Grid discretization parameters (pixel to tile conversion)
# These define the playable area within the image
IMAGE_WIDTH = 428
IMAGE_HEIGHT = 683
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

        print(f"Loading {len(self.files)} parquet files...")
        self.tables = []
        self.valid_indices_list = []
        
        for path in self.files:
            # Read table with memory mapping, NO filters to ensure mapping
            # We handle filtering via indices to avoid loading data into memory
            table = pq.read_table(path, memory_map=True)
            
            num_rows = table.num_rows
            valid_mask = np.ones(num_rows, dtype=bool)

            # 1. Filter by offset
            if "offset" in table.column_names:
                is_zero_offset = pc.equal(table.column("offset"), 0).to_numpy(zero_copy_only=False)
                valid_mask &= is_zero_offset
                # Remove offset column to match original schema (creates new table structure but shares data)
                table = table.drop(["offset"])
            
            # 2. Filter Spells
            if "card" in table.column_names:
                is_spell = pc.is_in(table.column("card"), value_set=pa.array(list(SPELL_CARDS)))
                is_not_spell = pc.invert(is_spell).to_numpy(zero_copy_only=False)
                valid_mask &= is_not_spell
            
            # 3. Add missing x and y columns with default value -1 if they don't exist
            if "x" not in table.column_names:
                # Insert after png_bytes
                png_bytes_idx = table.column_names.index("png_bytes") if "png_bytes" in table.column_names else 0
                x_col = pa.array([-1] * table.num_rows, type=pa.int16())
                table = table.add_column(png_bytes_idx + 1, "x", x_col)
            
            if "y" not in table.column_names:
                x_idx = table.column_names.index("x")
                y_col = pa.array([-1] * table.num_rows, type=pa.int16())
                table = table.add_column(x_idx + 1, "y", y_col)

            # Filter invalid placement (card != None AND x=-1, y=-1)
            c_col = table.column("card")
            x_col = table.column("x")
            y_col = table.column("y")
            
            is_played = pc.not_equal(c_col, "None")
            bad_pos = pc.and_(pc.equal(x_col, -1), pc.equal(y_col, -1))
            bad_rows = pc.and_(is_played, bad_pos)
            is_good_row = pc.invert(bad_rows).to_numpy(zero_copy_only=False)
            valid_mask &= is_good_row
            
            # 4. Filter out samples with invalid hand data
            if "hand" in table.column_names:
                # Get indices of rows that are valid so far
                current_indices = np.nonzero(valid_mask)[0]
                
                if len(current_indices) > 0:
                    # Only load 'hand' and 'card' for these rows to validate
                    hands = table.column("hand").take(current_indices).to_pylist()
                    cards = table.column("card").take(current_indices).to_pylist()
                    
                    kept_indices_local = []
                    
                    for i, (hand, card_played) in enumerate(zip(hands, cards)):
                         # Check if hand column has 4 entries
                        if len(hand) != 4:
                            continue
                        # Check if card played is in hand
                        if card_played != "None" and card_played not in hand:
                            continue
                        kept_indices_local.append(i)
                    
                    # Map local indices back to global indices
                    final_valid_indices = current_indices[kept_indices_local]
                else:
                    final_valid_indices = np.array([], dtype=np.int64)
            else:
                final_valid_indices = np.nonzero(valid_mask)[0]

            self.tables.append(table)
            self.valid_indices_list.append(final_valid_indices)
            print(f"File: {os.path.basename(path)} | Total: {num_rows} | Kept: {len(final_valid_indices)}")
        
        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.cumsum([len(inds) for inds in self.valid_indices_list])
        total_len = self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0
        print(f"Total samples loaded: {total_len}")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0

    def get_action_distribution(self):
        """Efficiently count positive (play card) and negative (no-op) samples."""
        num_negative = 0
        num_total = len(self)
        
        for table, indices in zip(self.tables, self.valid_indices_list):
            if len(indices) == 0:
                continue
                
            # "card" column contains the label
            # "None" is negative, everything else is positive
            if "card" in table.column_names:
                # Count occurrences of "None" in valid rows
                cards = table.column("card").take(indices)
                is_none = pc.equal(cards, "None")
                # sum() of boolean array gives count of True
                none_in_table = pc.sum(is_none).as_py()
                num_negative += none_in_table
        
        num_positive = num_total - num_negative
        return num_positive, num_negative

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Find which table contains the index
        table_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        if table_idx == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cumulative_lengths[table_idx - 1]

        # Retrieve row from the specific table
        table = self.tables[table_idx]
        real_row_idx = self.valid_indices_list[table_idx][row_idx]
        
        # Slice using the real index from the memory-mapped table
        row_table = table.slice(real_row_idx, 1)
        data = row_table.to_pydict()

        # Decode the image from bytes
        raw_image_bytes = data["png_bytes"][0]
        img = Image.open(io.BytesIO(raw_image_bytes))
        t = torch.from_numpy(np.array(img).astype(np.float32))
        # (H, W, C) -> (C, H, W)
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        frames = t.unsqueeze(0) # (1, C, H, W)

        frames = frames / 255.0

        # Parse card name and convert to ID
        card_name = data["card"][0]
        
        # Action label: 0 = no-op (None), 1 = play a card
        if card_name == "None":
            action = 0
            card_id = -1  # Will be ignored in card loss when action=0
        elif card_name in CARD_TO_ID:
            action = 1
            card_id = CARD_TO_ID[card_name]
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
            playable_mask = torch.ones(self.num_cards, dtype=torch.float32)
            hand_mask = torch.ones(self.num_cards, dtype=torch.float32)
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
            playable_mask = torch.zeros(self.num_cards, dtype=torch.float32)
            playable_mask[playable_mask_ids] = 1.0

            hand_mask_ids = torch.tensor(hand_mask_ids, dtype=torch.long)
            hand_mask = torch.zeros(self.num_cards, dtype=torch.float32)
            hand_mask[hand_mask_ids] = 1.0
        
        elixir = float(data["elixir"][0]) if "elixir" in data else 10.0

        # Normalize numeric features to [0, 1] range to avoid overwhelming LayerNorm
        # Elixir is 0-10
        norm_elixir = elixir / 10.0

        numeric_features = torch.tensor([
            norm_elixir
        ], dtype=torch.float32)
        label_action = torch.tensor(action, dtype=torch.long)
        label_card = torch.tensor(card_id, dtype=torch.long)
        label_placement = torch.tensor(tile_index, dtype=torch.long)

        return {
            "frames": frames,
            "playable_mask": playable_mask,
            "hand_mask": hand_mask,
            "numeric_features": numeric_features,
            "label_action": label_action,
            "label_card": label_card,
            "label_placement": label_placement,
        }