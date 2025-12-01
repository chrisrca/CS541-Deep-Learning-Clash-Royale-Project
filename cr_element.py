import cv2
import numpy as np

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

def crop_arena(image: np.ndarray) -> np.ndarray:
    arena_box = BoundingBox(origin=(57, 137), size=(428, 683))
    return image[arena_box.y_start():arena_box.y_end(), arena_box.x_start():arena_box.x_end()]