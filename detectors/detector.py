from detectors.yolo.yolo_detector import get_bounding_boxes as yolo_gbb
from detectors.haarc.hc_detector import get_bounding_boxes as hc_gbb
from detectors.bgsub.bgsub_detector import get_bounding_boxes as bgsub_gbb
from detectors.ssd.ssd import get_bounding_boxes as ssd_gbb
from detectors.tfoda.tfoda_detector import get_bounding_boxes as tfoda_gbb

def get_bounding_boxes(frame, model):
    if model == 'yolo':
        return yolo_gbb(frame)
    elif model == 'haarc':
        return hc_gbb(frame)
    elif model == 'bgsub':
        return bgsub_gbb(frame)
    elif model == 'ssd':
        return ssd_gbb(frame)
    elif model == 'tfoda':
        return tfoda_gbb(frame)
    else:
        raise Exception('Invalid detector model, algorithm or API specified (options: yolo, tfoda, haarc, bgsub, ssd)')