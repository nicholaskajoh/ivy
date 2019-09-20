def get_bounding_boxes(frame, model):
    if model == 'yolo':
        from detectors.yolo import get_bounding_boxes
    elif model == 'haarcascade':
        from detectors.haarcascade import get_bounding_boxes
    elif model == 'tfoda':
        from detectors.tfoda import get_bounding_boxes
    else:
        raise Exception('Invalid detector model, algorithm or API specified (options: yolo, tfoda, haarcascade)')

    return get_bounding_boxes(frame)