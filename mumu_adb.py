import subprocess
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable, Dict
import threading
from collections import deque
import numpy as np
import struct

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
    def __init__(self,
                 adb_path: str = "scrcpy/adb.exe",
                 port_range: Tuple[int, int] = (16000, 17000),
                 fps: int = 30):
        self.adb = str(Path(adb_path).resolve())
        self.start_port, self.end_port = port_range
        self.connected: List[str] = []
        self.fps = fps
        self._frame_queues: Dict[str, deque] = {}
        self._stream_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}

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

    def restart_adb(self) -> None:
        print("Killing & restarting ADB")
        _run(f'"{self.adb}" kill-server')
        _run(f'"{self.adb}" start-server')
        time.sleep(1.2)

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

    def _adb(self, serial: str, *args: str) -> str:
        cmd = [self.adb, "-s", serial] + list(args)
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

    def shell(self, serial: str, cmd: str) -> str:
        return self._adb(serial, "shell", cmd)

    def tap(self, serial: str, x: int, y: int) -> str:
        return self.shell(serial, f"input tap {x} {y}")

    def key(self, serial: str, keycode: str) -> str:
        return self.shell(serial, f"input keyevent {keycode}")

    def screenshot(self, serial: str, local_path: str) -> str:
        remote = "/sdcard/_tmp_screenshot.png"
        self.shell(serial, f"screencap {remote}")
        out = _run(f'"{self.adb}" -s {serial} pull {remote} "{local_path}"')
        self.shell(serial, f"rm {remote}")
        return out

    def run(self,
            func: Callable[..., Any],
            serials: Optional[str | List[str]] = None,
            max_workers: Optional[int] = None,
            **kwargs) -> List[Any] | Any:
        if serials is None:
            target = self.connected
        elif isinstance(serials, str):
            target = [serials]
        else:
            target = serials

        if len(target) == 1:
            return func(target[0], **kwargs)

        max_workers = max_workers or len(target)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            fut2ser = {exe.submit(func, s, **kwargs): s for s in target}
            for fut in as_completed(fut2ser):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    s = fut2ser[fut]
                    print(f"   [{s}] ERROR: {exc}")
                    results.append(exc)
        return results

    def _stream_worker(self, serial: str):
        """
        Optimized frame capture using raw framebuffer.
        Converts to match PNG color space for template compatibility.
        """
        stop = self._stop_events[serial]
        queue = self._frame_queues[serial]
        
        # Precompile command
        cmd = [self.adb, "-s", serial, "exec-out", "screencap"]
        
        frame_count = 0
        error_count = 0
        
        # Cache for dimensions
        cached_width = None
        cached_height = None
        
        while not stop.is_set() and error_count < 20:
            t0 = time.time()
            
            try:
                # Capture raw framebuffer
                raw = subprocess.check_output(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    timeout=0.5
                )
                
                if len(raw) > 12:
                    # Parse framebuffer header
                    if cached_width is None:
                        cached_width = struct.unpack('I', raw[0:4])[0]
                        cached_height = struct.unpack('I', raw[4:8])[0]
                    
                    # Convert RGBA framebuffer to BGR
                    expected_size = cached_width * cached_height * 4
                    pixels = np.frombuffer(raw[12:12+expected_size], dtype=np.uint8)
                    
                    if len(pixels) == expected_size:
                        # Reshape RGBA
                        frame_rgba = pixels.reshape((cached_height, cached_width, 4))
                        
                        # Convert to BGR (drop alpha channel, swap R and B)
                        # This matches how PNG decoding produces BGR
                        frame = frame_rgba[:, :, [2, 1, 0]]  # BGR order, no alpha
                        
                        # Ensure contiguous array for OpenCV
                        frame = np.ascontiguousarray(frame)
                        
                        # Update queue
                        if len(queue) >= 2:
                            queue.popleft()
                        queue.append(frame)
                        
                        frame_count += 1
                        error_count = 0
                    else:
                        error_count += 1
                
                else:
                    error_count += 1
            
            except subprocess.TimeoutExpired:
                error_count += 1
            except Exception as e:
                error_count += 1
                if error_count == 1:
                    print(f"[{serial}] Capture error: {e}")
            
            # Minimal sleep for rate limiting
            elapsed = time.time() - t0
            target_interval = 1.0 / self.fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def start_stream(self, serial: str) -> None:
        if serial in self._stream_threads:
            return
        
        self._frame_queues[serial] = deque(maxlen=5)
        self._stop_events[serial] = threading.Event()
        
        th = threading.Thread(
            target=self._stream_worker,
            args=(serial,),
            daemon=True,
            name=f"stream-{serial}"
        )
        th.start()
        self._stream_threads[serial] = th
        
        # Wait for first frame
        time.sleep(0.3)

    def stop_stream(self, serial: str) -> None:
        if serial not in self._stop_events:
            return
        
        self._stop_events[serial].set()
        
        if serial in self._stream_threads:
            self._stream_threads[serial].join(timeout=3)
        
        for attr in ("_frame_queues", "_stop_events", "_stream_threads"):
            getattr(self, attr).pop(serial, None)

    def get_screen(self, serial: str) -> Optional[np.ndarray]:
        if serial not in self._frame_queues:
            self.start_stream(serial)
        
        queue = self._frame_queues[serial]
        
        # Wait up to 2 seconds for first frame
        for _ in range(200):
            if queue:
                return queue[-1].copy()
            time.sleep(0.01)
        
        return None