def get_centroid(_bounding_box):
    (x, y, w, h) = _bounding_box
    return (round((x + x + w) / 2), round((y + y + h) / 2))

def box_contains_point(_bbox, _pt):
    return _bbox[0] < _pt[0] < _bbox[0] + _bbox[2] and _bbox[1] < _pt[1] < _bbox[1] + _bbox[3]


class Blob:
    def __init__(self, _bounding_box, _tracker):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0

    def update(self, _bounding_box, _tracker=None):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)
        if _tracker:
            self.tracker = _tracker