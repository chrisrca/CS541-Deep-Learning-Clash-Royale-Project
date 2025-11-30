import numpy as np
import cv2
import math

class CRElement:
    class BoundingBox:
        def __init__(self, origin: tuple, size: tuple):
            self.origin = origin
            self.size = size
            
        def x_start(self):
            return self.origin[0]
        def x_end(self):
            return self.x_start() + self.size[0]
        def y_start(self):
            return self.origin[1]
        def y_end(self):
            return self.y_start() + self.size[1]

    elixir = BoundingBox(origin=(103 - 84,823 + 97), size=(30,25))
    elixir_bar = BoundingBox(origin=(48,839+ 97), size=(418+54,3))
    arena = BoundingBox(origin=(57,137), size=(428,683))

    cards_in_hand = []
    cards_in_hand_match = []
    card_space = 5
    card_space_match = 13
    card = BoundingBox(origin=(86 - 14, 737 + 97), size=(66,81))
    card_match = BoundingBox(origin=(127, 801), size=(85, 106))
    for i in range(4):
        card_origin_offset = (card.origin[0] + (card.size[0] + card_space) * i, card.origin[1])
        cards_in_hand.append(BoundingBox(origin=card_origin_offset, size=card.size))
        if i == 2:
            card_space_match += 3
        card_origin_offset = (card_match.origin[0] + (card_match.size[0] + card_space_match) * i, card_match.origin[1])
        cards_in_hand_match.append(BoundingBox(origin=card_origin_offset, size=card_match.size))

    @staticmethod
    def is_grayscale(image):
        grayscaled = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return np.median(np.abs(image - grayscaled)) < 5 # If grayscale version (not ready to play)

    @staticmethod
    def cut_to_fit(image: np.ndarray, box: BoundingBox) -> np.ndarray:
        return image[box.y_start():box.y_end(),box.x_start():box.x_end()]
    
    @staticmethod
    def get_images_in_hand(screenshot: np.ndarray, is_match: bool = False) -> list[np.ndarray]:
        if is_match:
            cards = [CRElement.cut_to_fit(image=screenshot, box=bounding_box) for bounding_box in CRElement.cards_in_hand_match]
            cards = [cv2.resize(card, (66,81), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR) for card in cards] # TODO use bounding box
            return cards
        else:
            return [CRElement.cut_to_fit(image=screenshot, box=bounding_box) for bounding_box in CRElement.cards_in_hand]

    @staticmethod
    def get_elixir_pic(screenshot: np.ndarray) -> np.ndarray:
        return CRElement.cut_to_fit(screenshot, CRElement.elixir)
    
    @staticmethod
    def quantize(frame: np.ndarray, levels=4, display=False) -> np.ndarray:
        """ Deprecated. """
        step_size = 255 // (levels - 1) if display else 1
        quantized = np.clip((frame / 255 * (levels - 1)).astype(int), 0, levels - 1)
        quantized = (quantized * step_size).astype(np.uint8)
        
        return quantized
    
    @staticmethod
    def blur(frame: np.ndarray, kernel = (3,3)):
        return cv2.GaussianBlur(frame, ksize=kernel, sigmaX=0, sigmaY=0)
    
    @staticmethod
    def downsize(frame: np.ndarray, factor: int):
        return cv2.resize(frame, (frame.shape[1]//factor, frame.shape[0]//factor), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def not_ready_overlay(img: np.ndarray, current_elixir: float, cost: int, intensity: int = 127):
        """
        Add the grayscale loading overlay to an image similar to the in-game effect for when a card is too expensive to be played.
        Parameters:
            img (np.ndarray) : Base BGR or grayscale image.
            current_elixir (float) : Current elixir
            cost (int) : Elixir cost
            intensity : Overlay intensity. 127 has been found to provide good results, do not change unless needed.

        Returns:
            np.ndarray : New BGR image with not ready effect
        """
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        radius = int(np.hypot(cx, cy))
        readiness = 1 - current_elixir / cost

        steps = max(10, int(100 * readiness))
        angles = np.linspace(-math.pi/2, -math.pi/2 - 2*math.pi*readiness, steps)

        # Create a blank slice mask
        slice_mask = np.zeros_like(img, dtype=np.uint8)

        # Build polygon points
        points = [(cx, cy)] + [
            (int(cx + radius * math.cos(a)), int(cy + radius * math.sin(a)))
            for a in angles
        ]
        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        is_color = len(slice_mask.shape) == 3

        # Fill the slice with the intensity
        if is_color:
            cv2.fillPoly(slice_mask, [pts], (intensity, intensity, intensity))
        else:  # grayscale
            cv2.fillPoly(slice_mask, [pts], intensity)

        result = cv2.add(img, slice_mask)
        if not is_color:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result
    
    @staticmethod
    def prep(image, downsize_factor = 2, blur_kernel: tuple[int,int] = (3,3)): # TODO never used this downsize factor... need to test and see if tuning helps
        shrunk = CRElement.downsize(image, downsize_factor)
        blurred = CRElement.blur(shrunk, blur_kernel) # TODO test different blur kernal
        return blurred
    
    @staticmethod
    def best_match(template: np.ndarray, labelled_image_collection : dict[str, np.ndarray], shearing: tuple[int, int] = (0,0), show=False) -> tuple[float, str]:
        """
        
        """
        best_score, best_match = 0, None
        shear_x, shear_y = shearing[1], shearing[0]
        sheared_template = template[shear_y:-shear_y, shear_x:-shear_x]
        if(show):
            cv2.imshow("template", sheared_template)
            cv2.waitKey(0)
        for name, image in labelled_image_collection.items():
            if type(image) != np.ndarray:
                image = image.BGR
            
            res = cv2.matchTemplate(image, sheared_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_match = name
                best_score = max_val
        if(show):
            cv2.imshow("match", labelled_image_collection[best_match])
            cv2.waitKey(0)
        return best_score, best_match
    
    class Card():
        def __init__(self, filename:str = None, image:np.ndarray = None, name:str = None):
            self.BGR = cv2.imread(f"cr_detection/cards/{filename}") if filename is not None else image
            self.GRAY = cv2.cvtColor(self.BGR, cv2.COLOR_BGR2GRAY)
            self.cost = int(filename.split("-")[1].split(".")[0]) if filename is not None else None
            self.name = name if name is not None else filename.split("-")[0]

    class Elixir():
        def __init__(self, filename: str):
            self.BGR = cv2.imread(f"cr_detection/elixir/{filename}")
            self.value = int(filename.split(".")[0])
    
