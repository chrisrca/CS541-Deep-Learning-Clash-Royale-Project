import numpy as np

OCR_CFG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#'

MENU_CHECK_REGION = [slice(200, 220), slice(420, 440)] # [slice(top_left_y, bottom_right_y), slice(top_left_x, bottom_right_x)]
MENU_TARGET_COLOR = np.array([16, 187, 248], dtype=np.float32) # BGR

HAMBURGER_MENU = (495, 105)
TV_ROYALE_BUTTON = (390, 210)

TV_ROYALE_LEFT = (46, 166)
TV_ROYALE_RIGHT = (495, 167)
ARENA_REGION = [slice(145, 210),slice(85, 460)]
ARENA_NAME = [slice(150, 183), slice(65, 482)]
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
