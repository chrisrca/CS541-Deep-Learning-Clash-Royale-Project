import time
from pathlib import Path
from mumu_adb import MuMuADB

mumu = MuMuADB(adb_path="scrcpy/adb.exe")
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)

def record_and_do_stuff(serial: str):
    # Remote storage path
    remote = f"/sdcard/replay_{serial.split(':')[1]}_{int(time.time())}.mp4"
    
    # Start recording
    proc = mumu.start_recording(serial, remote)
    
    # Tap at (100, 100), wait 10s, press HOME
    mumu.tap(serial, 100, 100)
    time.sleep(10)
    mumu.key(serial, "KEYCODE_HOME")

    # Stop recording and pull file
    mumu.stop_recording(serial, proc)
    local = Path("recordings") / Path(remote).name
    local.parent.mkdir(parents=True, exist_ok=True)
    mumu.pull_recording(serial, remote, str(local))
    print(f"Finished {serial}: {local}")
    return str(local)

def take_screenshot(serial: str) -> Path:
    Path("screenshots").mkdir(exist_ok=True)
    local_path = Path("screenshots") / f"screenshot_{serial.replace(':', '_')}.png"
    mumu.screenshot(serial, str(local_path))
    return local_path

# Run on all devices
saved_all = mumu.run(record_and_do_stuff)

time.sleep(2)

# Run on just the first device
saved_one = mumu.run(record_and_do_stuff, serials[0])

time.sleep(2)

# Run on devices 0 and 2 only
saved_two = mumu.run(record_and_do_stuff, [serials[0], serials[2]])

time.sleep(2)

# Screenshot on first device
screenshot_path = mumu.run(take_screenshot, serials=serials[0])