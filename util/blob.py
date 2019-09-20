from .bounding_box import get_centroid, get_area


class Blob:
    def __init__(self, _bounding_box, _type, _confidence, _tracker):
        self.bounding_box = _bounding_box
        self.type = _type
        self.type_confidence = _confidence
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0
        self.num_consecutive_detection_failures = 0
        self.counted = False
        self.position_first_detected = tuple(self.centroid)

    def update(self, _bounding_box, _type=None, _confidence=None, _tracker=None):
        self.bounding_box = _bounding_box
        self.type = _type if _type != None else self.type
        self.type_confidence = _confidence if _confidence != None else self.type_confidence
        self.centroid = get_centroid(_bounding_box)
        self.area = get_area(_bounding_box)
        if _tracker:
            self.tracker = _tracker