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

# Load elixir & card templates
elixir_templates = [CRElement.Elixir(f"{i}.png") for i in range(11)]
card_templates = {CRElement.Card(filename).name: CRElement.Card(filename) for filename in os.listdir("cr_detection/cards/")}

def play_game(serial: str):
    # Track previous line length
    prev_len = 0

    while True:
        frame = mumu.get_screen(serial)
        if frame is None:
            continue

        # Get current elixir
        elixir = CRGameState.current_elixir(
            screenshot=frame,
            elixir_images=elixir_templates,
            confidence=0.7,
            is_match=True
        )

        # Get cards in hand
        cards = CRGameState.cards_in_hand(
            screenshot=frame,
            card_images=card_templates,
            current_elixir=elixir,
            confidence_color=0.6,
            confidence_gray=0.55,
            is_match=True
        )

        # Print status
        elixir_str = f"{elixir:5.2f}" if elixir is not None else "-.--"
        cards_str = " | ".join([card if card else "empty" for card in cards])
        
        output = f"Elixir: {elixir_str} | Cards: {cards_str}"
        
        # Add spaces only if current output is shorter than previous
        if len(output) < prev_len:
            output += " " * (prev_len - len(output))
        prev_len = len(output)
        
        print(f"\r{output}", end="", flush=True)

        time.sleep(0.08)

mumu.run(play_game)