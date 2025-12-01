import numpy as np

OCR_CFG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#'

MENU_CHECK_REGION = [slice(200, 220), slice(420, 440)] # [slice(top_left_y, bottom_right_y), slice(top_left_x, bottom_right_x)]
MENU_TARGET_COLOR = np.array([16, 187, 248], dtype=np.float32) # BGR

HAMBURGER_MENU = (495, 105)
TV_ROYALE_BUTTON = (390, 210)
TRAINING_CAMP_BUTTON = (390, 300)

TV_ROYALE_LEFT = (46, 166)
TV_ROYALE_RIGHT = (495, 167)
ARENA_REGION = [slice(145, 210),slice(86, 461)]
WATCHED_INDICATOR_REGION = [slice(233, 255), slice(229, 303)]
WATCHED_TARGET_COLOR = np.array([155, 252, 106], dtype=np.float32)
WATCH_BUTTON = (453, 446)
                        
REPLAY_TOP_COLOR_REGION = [slice(1, 6), slice(67, 409)]
REPLAY_BOTTOM_COLOR_REGION = [slice(954, 960), slice(451, 520)]
REPLAY_BOTTOM_COLOR = np.array([151, 90, 58], dtype=np.float32)
REPLAY_TOP_COLOR = np.array([58, 69, 149], dtype=np.float32)
REPLAY_OK_BUTTON_REGION = [slice(850, 895), slice(210, 336)]
REPLAY_OK_BUTTON_COLOR = np.array([244, 178, 97], dtype=np.float32)
REPLAY_OK_BUTTON = (272, 874)

TV_ROYALE_BANNER_COLOR_REGION = [slice(150, 185), slice(90, 100)]
TV_ROYALE_BANNER_COLOR = np.array([99, 26, 232], dtype=np.float32)

MENU_OK_BUTTON = (370, 550)

BATTLE_REGION = [slice(953, 957), slice(205, 525)]
BATTLE_TARGET_COLOR = np.array([137, 67, 6], dtype=np.float32)
BATTLE_CARD_1 = (170, 850)
BATTLE_CARD_2 = (270, 850)
BATTLE_CARD_3 = (370, 850)
BATTLE_CARD_4 = (470, 850)
BATTLE_PLACE_REGION = [slice(140, 630), slice(90, 450)]

ALL_CARDS = [
    "archer_queen", "archers", "arrows", "baby_dragon", "balloon", "bandit", "barb_barrel", "barb_hut", "barbs", 
    "bats", "battle_ram", "berserker", "bomb_tower", "bomber", "boss_bandit", "bowler", "bush_goblin", 
    "caged_goblin", "cannon", "cannon_cart", "clone", "dark_prince", "dart_goblin", "e_barbs", "e_wiz", 
    "earthquake", "electro_dragon", "electro_giant", "electro_spirit", "elixir_golem", "elixir_pump", "empty", 
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