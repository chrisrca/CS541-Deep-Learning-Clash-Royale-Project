import numpy as np
import cv2
from cr_element import CRElement

class CRGameState:
    
    @staticmethod
    def _count_proportion_matching_pixels(image: np.ndarray, target_color: np.ndarray, tolerance: int, max: int = None) -> float:
        """
        Parameters:
            image : BGR image
            target_color : BGR pixel value
        Returns:
            float: Estimated elixir between whole-number increments.
        """
        if max is None:
            max = image.shape[0] * image.shape[1]

        # Compute absolute difference per channel
        diff = np.abs(image - target_color)

        # Mask of pixels within tolerance in all 3 channels
        mask = np.all(diff <= tolerance, axis=-1)

        # Count pixels that match
        count = np.count_nonzero(mask)
        return count/max
    
    @staticmethod
    def _fractional_elixir(screenshot: np.ndarray):
        """
        Returns:
            float: Estimated elixir between whole-number increments.
        """
        screenshot = CRElement.blur(screenshot, (3,3))
        elixir_bar = CRElement.cut_to_fit(screenshot, CRElement.elixir_bar)

        return CRGameState._count_proportion_matching_pixels(
            image=elixir_bar,
            target_color=np.array([132, 71, 78]),
            tolerance=11,
            max=135
        )
    
    @staticmethod
    def is_card_selected(card_image: np.ndarray, confidence: float = .4):
        card_image = CRElement.blur(card_image, (5,5))
        bottom_of_card = card_image[75:,:]

        proportion_matching_pixels = CRGameState._count_proportion_matching_pixels(
            image=bottom_of_card,
            target_color=np.array([133, 73, 47]),
            tolerance=11
        )
        return proportion_matching_pixels > confidence
    
    @staticmethod
    def is_card_sliding(card_image: np.ndarray, confidence: float = .35):
        card_image = CRElement.blur(card_image, (5,5))
        left_of_card = card_image[:,5:7]
        right_of_card = card_image[:,-7:-5]
        proportion_matching_pixels_left = CRGameState._count_proportion_matching_pixels(
            image=left_of_card,
            target_color=np.array([133, 73, 47]),
            tolerance=20
        )
        proportion_matching_pixels_right = CRGameState._count_proportion_matching_pixels(
            image=right_of_card,
            target_color=np.array([133, 73, 47]),
            tolerance=20
        )
        return max(proportion_matching_pixels_left, proportion_matching_pixels_right) > confidence
    
    @staticmethod
    def current_elixir(screenshot: np.ndarray, elixir_images: list[CRElement.Elixir], confidence: float = 0.7):    
        best_score, best_match = CRElement.best_match(
            template=CRElement.get_elixir_pic(screenshot),
            labelled_image_collection={elixir.value: elixir.BGR for elixir in elixir_images},
            shearing=(2,2)
        )

        if(best_score > confidence):
            return best_match + CRGameState._fractional_elixir(screenshot)
        else:
            return None
    
    @staticmethod
    def cards_in_hand(screenshot: np.ndarray, card_images: dict[str, CRElement.Card], current_elixir: float, confidence_color: float = 0.6, confidence_gray = 0.55) -> list[CRElement.Card]:    
        cards_h = CRElement.get_images_in_hand(screenshot)
        detected_hand = []
        for card_h in cards_h:
            is_gray = CRElement.is_grayscale(card_h)
            if(CRGameState.is_card_sliding(card_h)):
                detected_hand.append(None)
                continue
            labelled_image_collection = None
            if(is_gray and current_elixir is not None):
                # card_images = [card for card in card_images if card.cost > current_elixir + buffer] # This may have perfomance improvements but may also miss cards...
                labelled_image_collection = {card.name: CRElement.not_ready_overlay(card.GRAY, current_elixir, card.cost) for card in card_images.values()}
            else:
                labelled_image_collection = {card.name: card.BGR for card in card_images.values()}

            labelled_image_collection = {name: CRElement.prep(image) for (name, image) in labelled_image_collection.items()} # TODO pre-prep images in card init for better performance?
            
            best_score, best_match = CRElement.best_match(
                template=CRElement.prep(card_h), # TODO test different prepping downscaling for gray & non-gray (1,4)
                labelled_image_collection=labelled_image_collection,
                shearing=(9,3)
            )
            if(is_gray and best_score > 0.55):
                detected_hand.append(f"gray_{best_match}")
            elif(best_score > 0.6):
                detected_hand.append(best_match)
            else:
                detected_hand.append(None)
        return detected_hand