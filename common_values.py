import numpy as np

OCR_CFG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#'

MENU_CHECK_REGION = [slice(200, 220), slice(420, 440)]
MENU_TARGET_COLOR = np.array([16, 187, 248], dtype=np.float32)  # BGR

HAMBURGER_MENU = (495, 105)
TV_ROYALE_BUTTON = (390, 210)

TV_ROYALE_LEFT = (46, 166)
TV_ROYALE_RIGHT = (495, 167)
ARENA_NAME = [slice(150, 183), slice(65, 482)]
WATCHED_INDICATOR_REGION = [slice(233, 255), slice(229, 303)]
WATCHED_TARGET_COLOR = np.array([155, 252, 106], dtype=np.float32)