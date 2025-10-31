### Basic Setup:
1. Install python [3.9+](https://www.python.org/downloads/)
2. Install [adb](https://developer.android.com/tools/adb) and ensure it is added to your PATH
3. Install [mumuplayer](https://www.mumuplayer.com/) and create **n** devices

With MuMu devices running, run `mumu.py`. Example output:
```ps1
python mumu.py
Killing & restarting ADB server...

Detecting MuMu ADB ports...
    Scan done in 3.21s, 2 port(s) open

Connecting...
    Connected to 127.0.0.1:16416
    Connected to 127.0.0.1:16448

Sending HOME key...
    HOME sent to 127.0.0.1:16416
    HOME sent to 127.0.0.1:16448
```