import time
import numpy as np
import cv2
import pytesseract
from difflib import SequenceMatcher
from threading import Lock
from mumu_adb import MuMuADB
from common_values import (
    ARENA_NAME, 
    HAMBURGER_MENU, 
    MENU_CHECK_REGION, 
    MENU_TARGET_COLOR,
    OCR_CFG, 
    TV_ROYALE_BUTTON,
    TV_ROYALE_LEFT, 
    TV_ROYALE_RIGHT,
    WATCHED_INDICATOR_REGION,
    WATCHED_TARGET_COLOR
)

mumu = MuMuADB(adb_path="scrcpy/adb.exe", fps=60)
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)

arena_names_global = []
collection_done = False
collection_lock = Lock()
first_serial = None

def open_clash_royale(serial: str):
    mumu.shell(serial, "am force-stop com.supercell.clashroyale")
    res = mumu.shell(serial, "am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
    print(f"[{serial}] {res.splitlines()[0]}")

def wait_for_menu(serial: str):
    while True:
        frame = mumu.get_screen(serial)
        if frame is None:
            continue
        roi = frame[*MENU_CHECK_REGION]
        mean = np.mean(roi, axis=(0, 1)).astype(np.float32)
        if np.all(np.abs(mean - MENU_TARGET_COLOR) <= 10):
            print(f"[{serial}] Detected Menu Screen")
            break

def open_tv_royale(serial: str):
    print(f"[{serial}] Opening TV Royale")
    mumu.tap(serial, *HAMBURGER_MENU)
    time.sleep(0.5)
    mumu.tap(serial, *TV_ROYALE_BUTTON)
    time.sleep(1)

def collect_arena_names(serial: str):
    GOBLIN_TARGET = "GoblinStadium"
    FUZZY_THRESHOLD = 0.75

    def is_goblin_stadium(text: str) -> bool:
        if not text:
            return False
        clean = ''.join(text.split())
        return SequenceMatcher(None, clean, GOBLIN_TARGET).ratio() >= FUZZY_THRESHOLD

    def preprocess_roi(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return cv2.resize(thresh, None, fx=2, fy=2)

    def get_name():
        frame = mumu.get_screen(serial)
        if frame is None:
            return None
        roi = frame[*ARENA_NAME]
        processed = preprocess_roi(roi)
        return pytesseract.image_to_string(processed, config=OCR_CFG).strip()

    global collection_done, arena_names_global, first_serial

    with collection_lock:
        if serial != first_serial:
            print(f"[{serial}] Not collector, searching for Goblin Stadium only...")

    # Search until Arena 1 then wait for collector
    if serial != first_serial:
        while True:
            name = get_name()
            if is_goblin_stadium(name):
                print(f"[{serial}] Found Goblin Stadium, waiting for collector to finish...")
                while not collection_done:
                    time.sleep(0.5)
                return arena_names_global
            mumu.tap(serial, *TV_ROYALE_RIGHT)
            time.sleep(1)

    # Arena collector
    print(f"[{serial}] Collector device, searching for Arena 1...")
    while True:
        name = get_name()
        if is_goblin_stadium(name):
            print(f"[{serial}] Found Arena 1, starting collection...")
            break
        mumu.tap(serial, *TV_ROYALE_RIGHT)
        time.sleep(1)

    arena_names = []
    while True:
        mumu.tap(serial, *TV_ROYALE_RIGHT)
        time.sleep(1)
        name = get_name()
        arena_names.append(name)
        if is_goblin_stadium(name):
            print(f"[{serial}] Total collected: {len(arena_names)}")
            with collection_lock:
                arena_names_global = arena_names
                collection_done = True
            break

    return arena_names

def jump_to_segment_start(serial: str):
    global arena_names_global, first_serial

    # Make sure collector has finished
    while not collection_done:
        time.sleep(0.1)

    total_arenas = len(arena_names_global)
    arenas_per_device = (total_arenas + len(serials) - 1) // len(serials)
    index = serials.index(serial)

    # Start of this device's segment
    target_arena_idx = index * arenas_per_device
    if target_arena_idx >= total_arenas:
        return

    print(f"[{serial}] Jumping to arena #{target_arena_idx + 1}")

    for _ in range(target_arena_idx):
        mumu.tap(serial, *TV_ROYALE_RIGHT)
        time.sleep(0.1)

def traverse_segment(serial: str):
    global arena_names_global, first_serial

    # Determine owned arenas
    total_arenas = len(arena_names_global)
    arenas_per_device = (total_arenas + len(serials) - 1) // len(serials)
    idx = serials.index(serial)
    segment_start_idx = idx * arenas_per_device
    segment_end_idx = min((idx + 1) * arenas_per_device, total_arenas)
    segment_size = segment_end_idx - segment_start_idx

    print(f"[{serial}] Traversing {segment_size} arena(s): {segment_start_idx + 1} to {segment_end_idx}")

    # Right from first to last in segment
    moves_right = segment_size - 1
    for _ in range(moves_right):
        mumu.tap(serial, *TV_ROYALE_RIGHT)
        time.sleep(1)

    # Left to return to start
    for _ in range(moves_right):
        mumu.tap(serial, *TV_ROYALE_LEFT)
        time.sleep(0.1)

    print(f"[{serial}] Finished traverse & returned to arena #{segment_start_idx + 1}")

def check_if_watched(serial: str):
    frame = mumu.get_screen(serial)
    roi = frame[*WATCHED_INDICATOR_REGION]

    # Compute absolute difference
    diff = np.abs(roi.astype(np.int16) - WATCHED_TARGET_COLOR)
    matches = np.all(diff <= 10, axis=-1)

    # Count matching pixels
    match_count = np.sum(matches)

    return match_count >= 10

def worker(serial: str):
    global first_serial
    with collection_lock:
        if first_serial is None:
            first_serial = serial
            print(f"[{serial}] Assigned as arena collector")
    
    open_clash_royale(serial)
    wait_for_menu(serial)
    open_tv_royale(serial)
    collect_arena_names(serial)
    jump_to_segment_start(serial)
    traverse_segment(serial)

    print(check_if_watched(serial))
    
    # cv2.imwrite(f"tv_royale_{serial.replace(':', '_')}.png", mumu.get_screen(serial))

mumu.run(worker)