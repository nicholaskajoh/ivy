from blobs.utils import get_centroid


class Blob:
    def __init__(self, _bbox, _roi_hist):
        self.bbox = _bbox
        self.roi_box = _bbox
        self.roi_hist = _roi_hist
        self.centroid = get_centroid(_bbox)