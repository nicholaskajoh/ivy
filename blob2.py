def get_centroid(_bounding_box):
    (x, y, w, h) = _bounding_box
    return (round((x + x + w) / 2), round((y + y + h) / 2))


class Blob:
    def __init__(self, _bounding_box, _tracker):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0

    def update(self, _bounding_box):
        self.bounding_box = _bounding_box
        self.centroid = get_centroid(_bounding_box)