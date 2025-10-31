import subprocess
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable

def _run(cmd: str) -> str:
    """Run a shell command, return stripped stdout or error."""
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
                 port_range: Tuple[int, int] = (16000, 17000)):
        self.adb = str(Path(adb_path).resolve())
        self.start_port, self.end_port = port_range
        self.connected: List[str] = [] # list of "127.0.0.1:xxxxx"

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

    # Note: I've read this can only go up to 3 minutes so maybe we start recording again once one is done and tack them together with ffmpeg
    def start_recording(self,
                        serial: str,
                        remote_path: str,
                        bitrate: str = "8000000",
                        size: str = "1080x1920") -> subprocess.Popen:
        """
        Start `screenrecord` in the background.

        Returns the Popen object so you can .kill() it later if needed.
        """
        cmd = [
            self.adb, "-s", serial, "shell", "screenrecord",
            "--bit-rate", bitrate,
            "--size", size,
            remote_path
        ]
        print(f"   [{serial}] Recording {remote_path}")
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop_recording(self, serial: str, proc: subprocess.Popen) -> None:
        """Graceful stop via SIGINT; fallback to kill."""
        print(f"   [{serial}] Stopping recording")
        self.shell(serial, "pkill -INT screenrecord")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def pull_recording(self, serial: str, remote: str, local: str) -> str:
        out = _run(f'"{self.adb}" -s {serial} pull {remote} "{local}"')
        print(f"   [{serial}] Pulled {Path(local).name}")
        # Cleanup
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