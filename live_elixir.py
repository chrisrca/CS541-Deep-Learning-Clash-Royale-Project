import cv2
import time
import os
from cr_detection.cr_element import CRElement
from cr_detection.cr_gamestate import CRGameState
from mumu_adb import MuMuADB

mumu = MuMuADB(adb_path="scrcpy/adb.exe")
mumu.restart_adb()
ports = mumu.scan_ports()
serials = mumu.connect_all(ports)
serial = serials[0]

elixir_templates = [CRElement.Elixir(f"{i}.png") for i in range(11)]

while True:
    frame = mumu.get_screen(serial)
    elixir = CRGameState.current_elixir(
        screenshot=frame,
        elixir_images=elixir_templates,
        confidence=0.7
    )

    if elixir is not None:
        print(f"\rElixir:{elixir:5.2f}", end="", flush=True)
    else:
        print(f"\rElixir: -.--", end="", flush=True)

    time.sleep(0.08)