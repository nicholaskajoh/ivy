def get_centroid(_bbox):
    (x, y, w, h) = _bbox
    return (round((x + x + w) / 2), round((y + y + h) / 2))


class Blob:
    def __init__(self, _bbox, _roi_hist):
        self.bbox = _bbox
        self.roi_box = _bbox
        self.roi_hist = _roi_hist
        self.centroid = get_centroid(_bbox)