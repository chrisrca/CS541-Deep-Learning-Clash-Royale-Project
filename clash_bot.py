import time
import cv2
import os
import random
import numpy as np
import torch

from decision_model.model import ConvLSTMClashRoyaleModel
from cr_detection.cr_element import CRElement
from cr_detection.cr_gamestate import CRGameState
from mumu_adb import MuMuADB
from common_values import *

# Load elixir templates
elixir_templates = [CRElement.Elixir(f"{i}.png") for i in range(11)]

# Load card templates
card_templates = {
    CRElement.Card(filename).name: CRElement.Card(filename) 
    for filename in os.listdir("cr_detection/cards/")
}


class ClashBot():
    def __init__(self):
        self.mumu = MuMuADB(adb_path="scrcpy/adb.exe", fps=30)
        self.mumu.restart_adb()
        self.ports = self.mumu.scan_ports()
        self.serials = self.mumu.connect_all(self.ports)
        # Track previous line length
        self.prev_len = 0
        self.delay = 10
        self.temperature = 0.7

        model = ConvLSTMClashRoyaleModel(
            num_cards=len(ALL_CARDS),
            numeric_feat_dim=1,
            grid_h=32,
            grid_w=18,
        )

        weights = torch.load("decision_model/checkpoints/best_model.pt", map_location=torch.device('cpu'))
        model.load_state_dict(weights['model_state_dict']) # My laptop doesn't have an NVIDIA GPU
        self.model = model
        self.model.eval()
        self.mumu.run(self.run)
    

    def run(self, serial):
        self.open_clash_royale(serial)
        while True:       
            # Loop starts on main menu  
            if not self.wait_for_menu(serial):
                self.open_clash_royale(serial)
                if not self.wait_for_menu(serial):
                    print(f"[{serial}] Menu unreachable - retrying in 60s")
                    time.sleep(60)
                    continue
   
            # Start a match
            self.open_training_camp(serial)
            self.wait_for_match(serial)
            # Play match until complete
            self.play(serial)
            break
            

            
    def play(self, serial):
        while not self.is_match_over(serial):
            frame = self.mumu.get_screen(serial)
    
            # Get current elixir
            elixir = CRGameState.current_elixir(
                screenshot=frame,
                elixir_images=elixir_templates,
                confidence=0.5
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
            if len(output) < self.prev_len:
                output += " " * (self.prev_len - len(output))
            self.prev_len = len(output)
            
            print(f"\r{output}", end="", flush=True)

            # GET MODEL OUTPUT
            # card_selected = output['card_logits'] (choose card from hand with highest logit value)
            # placement_logits = output['placement_logits'] (choose max value)

            # Hand mask is all 1s for cards in hand, 0s for empty slots
            # Playable mask is all 1s for cards that can be played with current elixir, 0s otherwise
            numeric_features = torch.tensor([elixir / 10.0 if elixir is not None else 1.0], dtype=torch.float32)
            hand_mask_array = [1 if card in cards else 0 for card in ALL_CARDS]
            hand_mask = torch.tensor(hand_mask_array, dtype=torch.float32)

            playable_mask_array = [1 for card in ALL_CARDS]
            playable_mask = torch.tensor(playable_mask_array, dtype=torch.float32)

            numeric_features = numeric_features.unsqueeze(0)  # Add batch dimension
            hand_mask = hand_mask.unsqueeze(0)  # Add batch dimension
            playable_mask = playable_mask.unsqueeze(0)  # Add batch dimension


            t = torch.from_numpy(frame.astype(np.float32))

            t = t.permute(2, 0, 1)
            frames = t[None, None, ...]
            frames = frames / 255.0


            with torch.no_grad():
                prediction = self.model(frames, numeric_features, playable_mask, hand_mask)
            
            # Action decision based on temperature
            action_logit = prediction['action_logits'][0].item()
            action_prob = torch.sigmoid(torch.tensor(action_logit)).item()
            
            should_act = action_prob > self.temperature
            choice = None
            
            if should_act:
                best_logit = -float('inf')
                
                # Find the best card to play among those in hand
                for card in cards:
                    if card is not None and card in ALL_CARDS:
                        index = ALL_CARDS.index(card)
                        card_logit = prediction['card_logits'][0, index].item()
                        
                        if card_logit > best_logit:
                            best_logit = card_logit
                            choice = card
            
            print(f"| Action Prob: {action_prob:.2f} | Act: {should_act} | Selected: {choice}")
            
            if choice is not None and choice in cards:
                selected_index = cards.index(choice)
                self.select_card(serial, selected_index + 1)
                x, y = random.randrange(BATTLE_PLACE_REGION[1].start, BATTLE_PLACE_REGION[1].stop), random.randrange(BATTLE_PLACE_REGION[0].start, BATTLE_PLACE_REGION[0].stop)
                self.place_card(serial, x, y)

            time.sleep(0.08)

    def screenshot(self, serial):
        frame = self.mumu.get_screen(serial)
        fp = f"screenshots/{time.time()}.png"
        cv2.imwrite(str(fp), frame)

    def open_clash_royale(self, serial: str):
        self.mumu.shell(serial, "am force-stop com.supercell.clashroyale")
        res = self.mumu.shell(serial, "am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
        print(f"[{serial}] {res.splitlines()[0]}")

    def wait_for_menu(self, serial: str, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame = self.mumu.get_screen(serial)
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

    def open_training_camp(self, serial: str):
        print(f"[{serial}] Opening Training Camp")
        self.mumu.tap(serial, *HAMBURGER_MENU)
        time.sleep(1)
        self.mumu.tap(serial, *TRAINING_CAMP_BUTTON)
        time.sleep(1)
        self.mumu.tap(serial, *MENU_OK_BUTTON)
    
    def wait_for_match(self, serial, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame = self.mumu.get_screen(serial)
            if frame is None:
                continue
            roi = frame[*BATTLE_REGION]
            mean = np.mean(roi, axis=(0, 1)).astype(np.float32)
            if np.all(np.abs(mean - BATTLE_TARGET_COLOR) <= 10):
                print(f"[{serial}] Match started")
                return True
            time.sleep(0.5)
        print(f"[{serial}] Match not detected")
        return False

    def is_match_over(self, serial):
        frame = self.mumu.get_screen(serial)
        if frame is None:
            return True
        roi = frame[*BATTLE_REGION]
        mean = np.mean(roi, axis=(0, 1)).astype(np.float32)
        if np.all(np.abs(mean - BATTLE_TARGET_COLOR) <= 10):
            return False
        return True
    
    def select_card(self, serial, card: int):
        if card == 1:
            self.mumu.tap(serial, *BATTLE_CARD_1)
        elif card == 2:
            self.mumu.tap(serial, *BATTLE_CARD_2)
        elif card == 3:
            self.mumu.tap(serial, *BATTLE_CARD_3)
        elif card == 4:
            self.mumu.tap(serial, *BATTLE_CARD_4)
        else:
            print(f"Invalid card selection: {card}")

    def place_card(self, serial, x: int, y: int):
        self.mumu.tap(serial, x, y)

ClashBot()