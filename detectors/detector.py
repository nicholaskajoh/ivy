from detectors.yolo.yolo_detector import get_bounding_boxes as yolo_gbb
from detectors.haarc.hc_detector import get_bounding_boxes as hc_gbb
from detectors.bgsub.bgsub_detector import get_bounding_boxes as bgsub_gbb

def get_bounding_boxes(frame, model):
    if model == 'yolo':
        return yolo_gbb(frame)
    elif model == 'haarc':
        return hc_gbb(frame)
    elif model == 'bgsub':
        return bgsub_gbb(frame)
    else:
        raise Exception('Invalid detector model/algorithm specified (options: yolo, haarc, bgsub)')