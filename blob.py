import cv2
import math

class Blob:
    def __init__(self, convex_hull):
        self.convex_hull = convex_hull
        x, y, w, h = cv2.boundingRect(convex_hull)
        self.bounding_rect = { 'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h }
        self.center_position = {
            'x': round((x + x + w) / 2),
            'y': round((y + y + h) / 2),
        }
        self.diagonal_size = math.sqrt(math.pow(w, 2) + math.pow(h, 2))
        self.aspect_ratio = float(w) / float(h)