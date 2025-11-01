from pathlib import Path
import time
import cv2
import numpy as np
import subprocess
from mumu_adb import MuMuADB

mumu = MuMuADB(adb_path="scrcpy/adb.exe")
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)

def open_clash_royale(serial: str):
    res = mumu.shell(serial, "am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
    print(f"[{serial}] {res.splitlines()[0]}")

def pull_screenshot_bytes(serial: str) -> bytes:
    remote = "/sdcard/_tmp.png"
    mumu.shell(serial, f"screencap {remote}")
    cmd = [mumu.adb, "-s", serial, "exec-out", f"cat {remote}"]
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        data = b""
    mumu.shell(serial, f"rm {remote}")
    return data

def wait_for_menu(serial: str):
    interval = 1.0 / 30
    target_color = np.array([16, 187, 248])
    y1, y2 = 200, 220
    x1, x2 = 420, 440

    while True:
        t0 = time.time()
        raw = pull_screenshot_bytes(serial)
        if not raw:
            time.sleep(max(0, interval - (time.time() - t0)))
            continue
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        if frame.shape[0] > y2 and frame.shape[1] > x2:
            roi = frame[y1:y2, x1:x2]
            mean_color = np.mean(roi, axis=(0,1))
            if np.all(np.abs(mean_color - target_color) <= 10):
                print(f"[{serial}] Detected Menu Screen")
                break

        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

def open_tv_royale(serial: str):
    print(f"[{serial}] Opening TV Royale")
    x, y = 495, 105
    mumu.tap(serial, x, y)
    time.sleep(0.5)
    x, y = 390, 210
    mumu.tap(serial, x, y)

def worker(serial: str):
    open_clash_royale(serial)
    wait_for_menu(serial)
    open_tv_royale(serial)

mumu.run(worker) 