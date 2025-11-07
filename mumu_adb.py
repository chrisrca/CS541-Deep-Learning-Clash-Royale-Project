import subprocess
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable, Dict
import threading
from collections import deque
import cv2
import numpy as np

def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.strip()}"

def _port_open(ip: str, port: int, timeout: float = 0.1) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0

class MuMuADB:
    def __init__(self, adb_path: str = "scrcpy/adb.exe",
                 port_range: Tuple[int, int] = (16000, 17000),
                 fps: int = 60):
        self.adb = str(Path(adb_path).resolve())
        self.start_port, self.end_port = port_range
        self.connected: List[str] = [] # list of "127.0.0.1:xxxxx"
        self.fps = fps
        self._frame_queues: Dict[str, deque] = {}
        self._stream_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}

    # Port scanning
    def scan_ports(self, start: Optional[int] = None,
                   end: Optional[int] = None) -> List[int]:
        start = start or self.start_port
        end = end or self.end_port
        ports = range(start, end + 1)
        print(f"Scanning {start}-{end}...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=64) as exe:
            results = exe.map(lambda p: (p, _port_open("127.0.0.1", p)), ports)
        open_ports = [p for p, ok in results if ok]
        print(f"Scanned in {time.time() - t0:.2f}s: {len(open_ports)} open")
        return open_ports

    # ADB server
    def restart_adb(self) -> None:
        print("Killing & restarting ADB")
        _run(f'"{self.adb}" kill-server')
        _run(f'"{self.adb}" start-server')
        time.sleep(1.2)

    # Connect
    def connect_all(self, ports: Optional[List[int]] = None) -> List[str]:
        if ports is None:
            ports = self.scan_ports()
        if not ports:
            raise RuntimeError("No MuMu instances detected.")
        serials = []
        for p in ports:
            s = f"127.0.0.1:{p}"
            out = _run(f'"{self.adb}" connect {s}')
            if "connected" in out.lower():
                serials.append(s)
                print(f"   Connected {s}")
            else:
                print(f"   Failed {s}: {out}")
        self.connected = serials
        return serials

    # Generic ADB wrappers (per serial)
    def _adb(self, serial: str, *args: str) -> str:
        """Build and run: adb -s <serial> <args…>"""
        cmd = [self.adb, "-s", serial] + list(args)
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

    def shell(self, serial: str, cmd: str) -> str:
        return self._adb(serial, "shell", cmd)

    def tap(self, serial: str, x: int, y: int) -> str:
        return self.shell(serial, f"input tap {x} {y}")

    def key(self, serial: str, keycode: str) -> str:
        return self.shell(serial, f"input keyevent {keycode}")

    def screenshot(self, serial: str, local_path: str) -> str:
        """Take a screenshot and pull it in one go."""
        remote = "/sdcard/_tmp_screenshot.png"
        self.shell(serial, f"screencap {remote}")
        out = _run(f'"{self.adb}" -s {serial} pull {remote} "{local_path}"')
        self.shell(serial, f"rm {remote}")
        return out

    def run(
        self,
        func: Callable[..., Any],
        serials: Optional[str | List[str]] = None,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> List[Any] | Any:
        """
        Universal runner - works for 1 device, many devices, or ALL devices.

        Args:
            serials: Single serial, list of serials, or None for all
            func: Function to run: func(serial: str, **kwargs) -> Any
            max_workers: Thread pool size (ignored for single device)
            **kwargs: Passed to every function call

        Returns:
            Single result (if 1 device) or List[results] (if multiple)
        """
        # Normalize serials input
        if serials is None:
            target_serials = self.connected
        elif isinstance(serials, str):
            target_serials = [serials]
        else:
            target_serials = serials

        # Single device
        if len(target_serials) == 1:
            return func(target_serials[0], **kwargs)

        # Multiple devices
        max_workers = max_workers or len(target_serials)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            future_to_serial = {
                exe.submit(func, s, **kwargs): s for s in target_serials
            }
            for future in as_completed(future_to_serial):
                try:
                    results.append(future.result())
                except Exception as exc:
                    serial = future_to_serial[future]
                    print(f"   [{serial}] ERROR: {exc}")
                    results.append(exc)

        return results

    # Streamer
    def _stream_worker(self, serial: str):
        interval = 1.0 / self.fps
        stop = self._stop_events[serial]
        queue = self._frame_queues[serial]
        
        # Use PNG format for consistency with template images
        cmd = [self.adb, "-s", serial, "exec-out", "screencap", "-p"]
        
        while not stop.is_set():
            t0 = time.time()
            
            try:
                raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                
                if raw:
                    arr = np.frombuffer(raw, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        if len(queue) >= 2:
                            queue.popleft()
                        queue.append(frame)
                        
            except subprocess.CalledProcessError:
                pass
            except Exception as e:
                print(f"[{serial}] Stream error: {e}")
            
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def start_stream(self, serial: str) -> None:
        if serial in self._stream_threads:
            return
        self._frame_queues[serial] = deque(maxlen=3)
        self._stop_events[serial] = threading.Event()
        th = threading.Thread(target=self._stream_worker, args=(serial,), daemon=True)
        th.start()
        self._stream_threads[serial] = th

    def stop_stream(self, serial: str) -> None:
        if serial not in self._stop_events:
            return
        self._stop_events[serial].set()
        self._stream_threads[serial].join(timeout=2)
        for k in ("_frame_queues", "_stop_events", "_stream_threads"):
            getattr(self, k).pop(serial, None)

    def get_screen(self, serial: str) -> Optional[np.ndarray]:
        if serial not in self._frame_queues:
            self.start_stream(serial)
        queue = self._frame_queues[serial]
        while True:
            if queue:
                return queue[-1]
            time.sleep(0.001)