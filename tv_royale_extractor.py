import os
import hashlib
import time
import shutil
import uuid
import cv2
import pytesseract
import queue
import numpy as np
from huggingface_hub import HfApi
from dotenv import load_dotenv
from pathlib import Path
from difflib import SequenceMatcher
from threading import Lock, Thread
from mumu_adb import MuMuADB
from common_values import (
    ARENA_NAME, 
    HAMBURGER_MENU, 
    MENU_CHECK_REGION, 
    MENU_TARGET_COLOR,
    OCR_CFG,
    REPLAY_BOTTOM_COLOR,
    REPLAY_BOTTOM_COLOR_REGION,
    REPLAY_OK_BUTTON,
    REPLAY_OK_BUTTON_COLOR,
    REPLAY_OK_BUTTON_REGION,
    REPLAY_TOP_COLOR,
    REPLAY_TOP_COLOR_REGION,
    TV_ROYALE_BANNER_COLOR,
    TV_ROYALE_BANNER_COLOR_REGION, 
    TV_ROYALE_BUTTON,
    TV_ROYALE_LEFT, 
    TV_ROYALE_RIGHT,
    WATCH_BUTTON,
    WATCHED_INDICATOR_REGION,
    WATCHED_TARGET_COLOR
)

REPLAY_ROOT = Path("replays")
REPLAY_ROOT.mkdir(exist_ok=True)

mumu = MuMuADB(adb_path="scrcpy/adb.exe", fps=60)
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
api = HfApi(token=HF_TOKEN)

REPO_ID = "chrisrca/clash-royale-tv-replays"

arena_names_global = []
collection_done = False
collection_lock = Lock()
first_serial = None

upload_queue = queue.Queue()

def upload_worker():
    while True:
        item = upload_queue.get()
        if item is None:
            break
        serial, replay_dir, arena_idx, replay_id = item
        try:
            rel_path = replay_dir.relative_to(REPLAY_ROOT)
            path_in_repo = rel_path.as_posix()
            api.upload_folder(
                folder_path=str(replay_dir),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                commit_message=f"Replay arena {arena_idx} {replay_id}",
                token=HF_TOKEN,
            )
            print(f"[{serial}] Uploaded {path_in_repo}")
            shutil.rmtree(replay_dir)
        except Exception as e:
            print(f"[{serial}] Upload failed: {e}")

uploader = Thread(target=upload_worker, daemon=True)
uploader.start()

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
        time.sleep(0.5)

def record_replays(serial: str):
    global arena_names_global

    total_arenas = len(arena_names_global)
    arenas_per_device = (total_arenas + len(serials) - 1) // len(serials)
    idx = serials.index(serial)
    segment_start_idx = idx * arenas_per_device
    segment_end_idx = min((idx + 1) * arenas_per_device, total_arenas)
    segment_size = segment_end_idx - segment_start_idx

    moves_right = segment_size - 1
    current_arena = segment_start_idx + 1

    # First arena (already on it)
    handle_replay(serial, current_arena)

    # Move right and check each subsequent arena
    for _ in range(moves_right):
        mumu.tap(serial, *TV_ROYALE_RIGHT)
        time.sleep(1)
        current_arena += 1
        handle_replay(serial, current_arena)
        time.sleep(1)

    # Return to start
    for _ in range(moves_right):
        mumu.tap(serial, *TV_ROYALE_LEFT)
        time.sleep(0.5)

    print(f"[{serial}] Finished traverse & returned to arena #{segment_start_idx + 1}")

def handle_replay(serial: str, arena_idx: int):
    def watched(serial: str):
        frame = mumu.get_screen(serial)
        roi = frame[*WATCHED_INDICATOR_REGION]

        # Compute absolute difference
        diff = np.abs(roi.astype(np.int16) - WATCHED_TARGET_COLOR)
        matches = np.all(diff <= 10, axis=-1)

        # Count matching pixels
        match_count = np.sum(matches)

        return match_count >= 10

    if not watched(serial):
        print(f"[{serial}] Arena {arena_idx} not watched yet!")
        mumu.tap(serial, *WATCH_BUTTON)

        replay_id = uuid.uuid4()
        replay_dir = REPLAY_ROOT / f"arena_{arena_idx}" / str(replay_id)
        replay_dir.mkdir(parents=True, exist_ok=True)
        frame_counter = 0
        last_frame_hash = None
        saved_frame_paths = []

        # Wait for replay to start
        while True:
            frame = mumu.get_screen(serial)
            if frame is None:
                continue
            top = frame[*REPLAY_TOP_COLOR_REGION]
            bottom = frame[*REPLAY_BOTTOM_COLOR_REGION]
            mean_top = (np.mean(top, axis=(0, 1)).astype(np.float32))
            mean_bottom = (np.mean(bottom, axis=(0, 1)).astype(np.float32))
            if np.all(np.abs(mean_top - REPLAY_TOP_COLOR) <= 10) and np.all(np.abs(mean_bottom - REPLAY_BOTTOM_COLOR) <= 10):
                print(f"[{serial}] Detected Replay")
                break

        # Wait for replay to end
        while True:
            frame = mumu.get_screen(serial)
            if frame is None:
                continue

            # Save frames here later or throw them onto hf (this is our "recording")
            current_hash = hashlib.md5(frame.tobytes()).hexdigest() # Ideally a clock would be better here but im tired so hash to detect frame changes
            if last_frame_hash is None or current_hash != last_frame_hash:
                frame_path = replay_dir / f"{(frame_counter - 10):05d}.png"
                if frame_counter > 10: # Skip first few frames to avoid replay speed / pause controls
                    cv2.imwrite(str(frame_path), frame)
                    saved_frame_paths.append(frame_path)  # Track saved file
                frame_counter += 1
                last_frame_hash = current_hash

            top = frame[*REPLAY_TOP_COLOR_REGION]
            bottom = frame[*REPLAY_BOTTOM_COLOR_REGION]
            button = frame[*REPLAY_OK_BUTTON_REGION]
            mean_top = (np.mean(top, axis=(0, 1)).astype(np.float32))
            mean_bottom = (np.mean(bottom, axis=(0, 1)).astype(np.float32))
            mean_button = (np.mean(button, axis=(0, 1)).astype(np.float32))
            if np.all(np.abs(mean_button - REPLAY_OK_BUTTON_COLOR) <= 10) and not (np.all(np.abs(mean_top - REPLAY_TOP_COLOR) <= 10) and np.all(np.abs(mean_bottom - REPLAY_BOTTOM_COLOR) <= 10)):
                print(f"[{serial}] Detected Replay End")
                break

        # Remove last 20 frames to avoid win / lose screen
        frames_to_remove = saved_frame_paths[-20:]
        for frame_path in frames_to_remove:
            if frame_path.exists():
                frame_path.unlink()
        
        if len(frames_to_remove) > 0:
            print(f"[{serial}] Removed {len(frames_to_remove)} trailing frames.")

        mumu.tap(serial, *REPLAY_OK_BUTTON)

        upload_queue.put((serial, replay_dir, arena_idx, replay_id))

        while True:
            frame = mumu.get_screen(serial)
            if frame is None:
                continue
            banner = frame[*TV_ROYALE_BANNER_COLOR_REGION]
            mean_banner = (np.mean(banner, axis=(0, 1)).astype(np.float32))
            if np.all(np.abs(mean_banner - TV_ROYALE_BANNER_COLOR) <= 10):
                print(f"[{serial}] Detected TV Royale")
                break

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
    while True:
        # Main replay recording loop
        record_replays(serial)
        time.sleep(600)

        # Relaunch app to make sure new replays become available (idk if this is necessary but with 6+ emulators time lost doing this doesn't really matter)
        open_clash_royale(serial)
        wait_for_menu(serial)
        open_tv_royale(serial)

    # cv2.imwrite(f"tv_royale_{serial.replace(':', '_')}.png", mumu.get_screen(serial))

mumu.run(worker)