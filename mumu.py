import subprocess, time, socket, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.strip()}"

def is_port_open(ip: str, port: int, timeout: float = 0.1) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0

# MuMu emulators are typically between ports 16000-17000 so just check them all
def fast_parallel_scan(start: int = 16000, end: int = 17000) -> list[int]:
    ports = list(range(start, end))
    with ThreadPoolExecutor(max_workers=32) as exe:
        results = exe.map(lambda p: (p, is_port_open("127.0.0.1", p)), ports)
    return [p for p, open_ in results if open_]

# Path to the custom adb.exe (scrcpy/adb.exe)
ADB_PATH = str(Path("scrcpy/adb.exe").resolve())

# Restart ADB
print("Killing & restarting ADB server...")
run(f'"{ADB_PATH}" kill-server')
run(f'"{ADB_PATH}" start-server')
time.sleep(1)

print("\nDetecting MuMu ADB ports...")
t0 = time.time()
open_ports = fast_parallel_scan()
print(f"   Scan done in {time.time()-t0:.2f}s, {len(open_ports)} port(s) open")

if not open_ports:
    print("   No MuMu instances found. Start players and retry.")
    sys.exit(1)

# Connect
print("\nConnecting...")
mumu_serials = []
for p in open_ports:
    ip_port = f"127.0.0.1:{p}"
    out = run(f'"{ADB_PATH}" connect {ip_port}')
    print(f"   Connected to {ip_port}")
    if "connected" in out:
        mumu_serials.append(ip_port)

# Send Home Button input
print("\nSending HOME key...")
def send_home(serial):
    run(f'"{ADB_PATH}" -s {serial} shell input keyevent KEYCODE_HOME')
    return serial

with ThreadPoolExecutor(max_workers=len(mumu_serials)) as exe:
    for s in exe.map(send_home, mumu_serials):
        print(f"   HOME sent to {s}")