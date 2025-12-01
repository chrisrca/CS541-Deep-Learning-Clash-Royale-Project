import glob
import hashlib
import os
import json
import shutil
import time
import uuid
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_commit, CommitOperationAdd
from threading import Thread
from pathlib import Path
from mumu_adb import MuMuADB
from common_values import (
    ARENA_REGION, 
    HAMBURGER_MENU, 
    MENU_CHECK_REGION, 
    MENU_TARGET_COLOR,
    REPLAY_BOTTOM_COLOR,
    REPLAY_BOTTOM_COLOR_REGION,
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
    WATCHED_TARGET_COLOR,
)

REPLAY_ROOT = Path("replays")
REPLAY_ROOT.mkdir(exist_ok=True)

mumu = MuMuADB(adb_path="scrcpy/adb.exe")
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
api = HfApi(token=HF_TOKEN)
REPO_ID = "chrisrca/clash-royale-tv-replays"

# Load arena templates: arena-1.png to arena-31.png
arena_files = sorted(
    glob.glob(os.path.join("arenas", "arena-*.png")),
    key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0])
)
TOTAL_ARENAS = len(arena_files)
print(f"Found {TOTAL_ARENAS} arena identifiers")

# Queue
QUEUE_FILE = REPLAY_ROOT / "upload_queue.jsonl"
QUEUE_FILE.touch(exist_ok=True)

def append_to_queue(serial, replay_dir, arena_idx, replay_id):
    item = {
        "serial": serial,
        "replay_dir": str(replay_dir),
        "arena_idx": arena_idx,
        "replay_id": str(replay_id),
        "timestamp": time.time()
    }
    with open(QUEUE_FILE, "a") as f:
        f.write(json.dumps(item) + "\n")
    total_parquets = sum(1 for _ in REPLAY_ROOT.rglob("*.parquet"))
    print(f"[{serial}] Queued {QUEUE_FILE.name} ({total_parquets} total)")

def upload_worker():
    while True:
        try:
            pending = []
            if QUEUE_FILE.exists():
                with open(QUEUE_FILE, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            replay_dir = Path(item["replay_dir"])
                            if replay_dir.exists():
                                pending.append((item["serial"], replay_dir, item["arena_idx"], item["replay_id"]))
                        except:
                            continue

            if not pending:
                print("[UPLOAD] Queue empty - sleeping 10 min")
                time.sleep(600)
                continue

            for _, replay_dir, arena_idx, replay_id in pending:
                print(f"[UPLOAD] Uploading {replay_id} from arena {arena_idx:02d}")
                try:
                    padded = f"arena_{arena_idx:02d}"
                    parquet_path = replay_dir / "frames.parquet"
                    preview_path = replay_dir / "preview.jpg"

                    operations = [
                        CommitOperationAdd(
                            path_or_fileobj=str(parquet_path.as_posix()),
                            path_in_repo=f"{padded}/{replay_id}/frames.parquet"
                        ),
                        CommitOperationAdd(
                            path_or_fileobj=str(preview_path.as_posix()),
                            path_in_repo=f"{padded}/{replay_id}/preview.jpg"
                        )
                    ]

                    create_commit(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        operations=operations,
                        commit_message=f"Replay {replay_id} (arena {arena_idx:02d})",
                        token=HF_TOKEN
                    )
                    print(f"[UPLOAD] SUCCESS: {replay_id}")
                    shutil.rmtree(replay_dir, ignore_errors=True)

                    # Remove from queue
                    remaining = []
                    for line in open(QUEUE_FILE):
                        if replay_id not in line:
                            remaining.append(line)
                    with open(QUEUE_FILE, "w") as f:
                        f.writelines(remaining)

                    time.sleep(5)
                except Exception as e:
                    print(f"[UPLOAD ERROR] {replay_id}: {e} — retrying in 60s")
                    time.sleep(60)
                    break

        except Exception as e:
            print(f"[UPLOAD CRITICAL] {e} — retrying in 60s")
            time.sleep(60)

upload_thread = Thread(target=upload_worker, daemon=True)
upload_thread.start()

def cleanup_old_replays():
    while True:
        time.sleep(1800)
        for p in REPLAY_ROOT.iterdir():
            if p.is_dir() and (time.time() - p.stat().st_mtime > 3600):
                shutil.rmtree(p, ignore_errors=True)
                print(f"[CLEANUP] Removed old: {p}")

cleanup_thread = Thread(target=cleanup_old_replays, daemon=True)
cleanup_thread.start()

def open_clash_royale(serial: str):
    mumu.shell(serial, "am force-stop com.supercell.clashroyale")
    res = mumu.shell(serial, "am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
    print(f"[{serial}] {res.splitlines()[0]}")

def wait_for_menu(serial: str, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        frame = mumu.get_screen(serial)
        if frame is None:
            time.sleep(0.1)
            continue
        roi = frame[*MENU_CHECK_REGION]
        mean = np.mean(roi, axis=(0, 1)).astype(np.float32)
        if np.all(np.abs(mean - MENU_TARGET_COLOR) <= 10):
            print(f"[{serial}] Detected Menu Screen")
            return True
        time.sleep(0.1)
    print(f"[{serial}] MENU TIMEOUT ({timeout}s)")
    return False

def open_tv_royale(serial: str):
    print(f"[{serial}] Opening TV Royale")
    mumu.tap(serial, *HAMBURGER_MENU)
    time.sleep(1)
    mumu.tap(serial, *TV_ROYALE_BUTTON)
    time.sleep(1.5)

def wait_for_tv_royale(serial: str, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        frame = mumu.get_screen(serial)
        if frame is None:
            continue
        banner = frame[*TV_ROYALE_BANNER_COLOR_REGION]
        mean = np.mean(banner, axis=(0, 1)).astype(np.float32)
        if np.all(np.abs(mean - TV_ROYALE_BANNER_COLOR) <= 10):
            print(f"[{serial}] In TV Royale")
            return True
        time.sleep(0.5)
    print(f"[{serial}] TV Royale not detected")
    return False

def detect_arena_index(serial: str, threshold=0.95) -> int:
    frame = mumu.get_screen(serial)
    if frame is None:
        return -1
    roi = frame[*ARENA_REGION]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_index = -1
    best_score = 0.0

    for file_path in arena_files:
        template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        result = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_index = int(os.path.basename(file_path).split('-')[1].split('.')[0])

    if best_score >= threshold:
        print(f"[{serial}] Detected Arena {best_index}")
        return best_index
    else:
        print(f"[{serial}] No arena match")
        return -1

def jump_to_assigned_arena(serial: str) -> bool:
    device_idx = serials.index(serial)
    arenas_per_device = (TOTAL_ARENAS + len(serials) - 1) // len(serials)
    target_start = device_idx * arenas_per_device + 1
    target_end = min(target_start + arenas_per_device - 1, TOTAL_ARENAS)

    if target_start > TOTAL_ARENAS:
        print(f"[{serial}] No arenas assigned")
        return False

    print(f"[{serial}] Assigned: Arenas {target_start} to {target_end}")

    current = detect_arena_index(serial)
    if current == -1:
        print(f"[{serial}] Failed to detect current arena")
        return False

    forward = (target_start - current) % TOTAL_ARENAS
    backward = (current - target_start) % TOTAL_ARENAS

    if forward == 0:
        print(f"[{serial}] Already on target arena {current}")
        return True

    if forward <= backward:
        taps, direction = forward, TV_ROYALE_RIGHT
        print(f"[{serial}] Moving RIGHT {taps} steps")
    else:
        taps, direction = backward, TV_ROYALE_LEFT
        print(f"[{serial}] Moving LEFT {taps} steps")

    for _ in range(taps):
        mumu.tap(serial, *direction)
        time.sleep(1)

    final = detect_arena_index(serial)
    if final == target_start:
        print(f"[{serial}] SUCCESS: Arrived at arena {target_start}")
    else:
        print(f"[{serial}] FAILED: Expected {target_start}, got {final}")
    return final == target_start

def is_watched(serial: str) -> bool:
    frame = mumu.get_screen(serial)
    if frame is None:
        return False
    roi = frame[*WATCHED_INDICATOR_REGION]
    diff = np.abs(roi.astype(np.int16) - WATCHED_TARGET_COLOR)
    matches = np.all(diff <= 10, axis=-1)
    match_count = np.sum(matches)
    watched = match_count >= 10
    print(f"[{serial}] Watched check: {'WATCHED' if watched else 'UNWATCHED'}")
    return watched

def traverse_and_find_unwatched(serial: str):
    device_idx = serials.index(serial)
    arenas_per_device = (TOTAL_ARENAS + len(serials) - 1) // len(serials)
    start_arena = device_idx * arenas_per_device + 1
    end_arena = min(start_arena + arenas_per_device - 1, TOTAL_ARENAS)

    print(f"[{serial}] Traversing arenas {start_arena} to {end_arena}")

    current_arena = start_arena
    max_steps = arenas_per_device

    for _ in range(max_steps):
        if current_arena > TOTAL_ARENAS:
            print(f"[{serial}] Reached end of arenas")
            break

        if not is_watched(serial):
            print(f"[{serial}] Should watch arena {current_arena}")
            return current_arena

        print(f"[{serial}] Arena {current_arena} already watched")
        
        if current_arena < end_arena:
            mumu.tap(serial, *TV_ROYALE_RIGHT)
            time.sleep(1)
            current_arena += 1
        else:
            break

    print(f"[{serial}] All arenas in segment watched")
    return None

def wait_for_replay_start(serial: str, timeout=60) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        frame = mumu.get_screen(serial)
        if frame is None:
            time.sleep(0.1)
            continue
        top = frame[*REPLAY_TOP_COLOR_REGION]
        bottom = frame[*REPLAY_BOTTOM_COLOR_REGION]
        mean_top = np.mean(top, axis=(0, 1)).astype(np.float32)
        mean_bottom = np.mean(bottom, axis=(0, 1)).astype(np.float32)
        if np.all(np.abs(mean_top - REPLAY_TOP_COLOR) <= 10) and np.all(np.abs(mean_bottom - REPLAY_BOTTOM_COLOR) <= 10):
            print(f"[{serial}] Detected Replay Start")
            return True
        time.sleep(0.1)
    print(f"[{serial}] REPLAY START TIMEOUT ({timeout}s)")
    return False

def record_and_queue_replay(serial: str, arena_idx: int):
    replay_id = uuid.uuid4()
    replay_dir = REPLAY_ROOT / f"arena_{arena_idx}" / str(replay_id)
    replay_dir.mkdir(parents=True, exist_ok=True)

    frame_counter = 0
    last_hash = None
    frames_buffer = []

    start_time = time.time()
    RECORD_TIMEOUT = 7 * 60

    print(f"[{serial}] Recording arena {arena_idx}")

    while time.time() - start_time < RECORD_TIMEOUT:
        frame = mumu.get_screen(serial)

        cur_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if last_hash is None or cur_hash != last_hash:
            if frame_counter > 40:
                _, png_bytes = cv2.imencode('.png', frame)
                frames_buffer.append({
                    "frame_id": frame_counter - 40,
                    "image": {
                        "bytes": png_bytes.tobytes(),
                        "path": f"frame_{frame_counter - 40:05d}.png"
                    },
                    "hash": cur_hash
                })
            frame_counter += 1
            last_hash = cur_hash

        top = frame[*REPLAY_TOP_COLOR_REGION]
        bot = frame[*REPLAY_BOTTOM_COLOR_REGION]
        btn = frame[*REPLAY_OK_BUTTON_REGION]
        mt = np.mean(top, axis=(0, 1)).astype(np.float32)
        mb = np.mean(bot, axis=(0, 1)).astype(np.float32)
        mbn = np.mean(btn, axis=(0, 1)).astype(np.float32)

        if np.all(np.abs(mbn - REPLAY_OK_BUTTON_COLOR) <= 10) and not (
            np.all(np.abs(mt - REPLAY_TOP_COLOR) <= 10)
            and np.all(np.abs(mb - REPLAY_BOTTOM_COLOR) <= 10)
        ):
            print(f"[{serial}] Detected Replay End")
            break

        time.sleep(0.05)

    else:
        print(f"[{serial}] RECORD TIMEOUT after {RECORD_TIMEOUT}s")
        shutil.rmtree(replay_dir, ignore_errors=True)
        return

    frames_buffer = frames_buffer[:-40] if len(frames_buffer) > 40 else frames_buffer

    if not frames_buffer:
        print(f"[{serial}] No frames recorded - discarding")
        shutil.rmtree(replay_dir, ignore_errors=True)
        return

    # Save preview jpg
    first_image_struct = frames_buffer[0]["image"]
    first_bytes = first_image_struct["bytes"]
    first_img = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(replay_dir / "preview.jpg"), first_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Write parquet
    table = pa.Table.from_pylist(frames_buffer)
    parquet_path = replay_dir / "frames.parquet"
    pq.write_table(
        table,
        str(parquet_path),
        compression="zstd",
        use_dictionary=True,
        write_statistics=True
    )
    print(f"[{serial}] Saved {len(frames_buffer)} frames {parquet_path.name} ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # Queue
    append_to_queue(serial, replay_dir, arena_idx, replay_id)
    print(f"[{serial}] Queued Parquet + preview upload for arena {arena_idx}")

def watch_replay(arena_index: int, serial: str):
    print(f"[{serial}] Starting to watch arena {arena_index}")
    
    mumu.tap(serial, *WATCH_BUTTON)
    time.sleep(1)
    
    if not wait_for_replay_start(serial):
        print(f"[{serial}] Replay failed to start")
        return
    
    print(f"[{serial}] Replay for arena {arena_index} is now playing.")
    record_and_queue_replay(serial, arena_index)

    print(f"[{serial}] Finished processing arena {arena_index}")

def worker(serial: str):
    while True:
        open_clash_royale(serial)

        if not wait_for_menu(serial):
            open_clash_royale(serial)
            if not wait_for_menu(serial):
                print(f"[{serial}] Menu unreachable - retrying in 60s")
                time.sleep(60)
                continue

        open_tv_royale(serial)
        if not wait_for_tv_royale(serial):
            print(f"[{serial}] TV Royale failed - retrying in 30s")
            time.sleep(30)
            continue

        if not jump_to_assigned_arena(serial):
            print(f"[{serial}] Jump failed - retrying in 30s")
            time.sleep(30)
            continue

        print(f"[{serial}] Starting traversal for unwatched replays")
        arena_to_watch = traverse_and_find_unwatched(serial)

        if arena_to_watch is None:
            print(f"[{serial}] All replays watched in segment - sleeping 10 minutes")
            time.sleep(600)
            continue
        else:
            watch_replay(arena_to_watch, serial)
            time.sleep(30)

mumu.run(worker)