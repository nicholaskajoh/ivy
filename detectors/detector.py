def get_bounding_boxes(frame, model):
    if model == 'yolo':
        from detectors.yolo.yolo_detector import get_bounding_boxes as yolo_gbb
        return yolo_gbb(frame)
    elif model == 'haarc':
        from detectors.haarc.hc_detector import get_bounding_boxes as hc_gbb
        return hc_gbb(frame)
    elif model == 'bgsub':
        from detectors.bgsub.bgsub_detector import get_bounding_boxes as bgsub_gbb
        return bgsub_gbb(frame)
    elif model == 'ssd':
        from detectors.ssd.ssd import get_bounding_boxes as ssd_gbb
        return ssd_gbb(frame)
    elif model == 'tfoda':
        from detectors.tfoda.tfoda_detector import get_bounding_boxes as tfoda_gbb
        return tfoda_gbb(frame)
    else:
        raise Exception('Invalid detector model, algorithm or API specified (options: yolo, tfoda, haarc, bgsub, ssd)')