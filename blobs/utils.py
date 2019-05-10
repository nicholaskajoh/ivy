def get_centroid(bounding_box):
    (x, y, w, h) = bounding_box
    return (round((x + x + w) / 2), round((y + y + h) / 2))

def box_contains_point(bbox, pt):
    return bbox[0] < pt[0] < bbox[0] + bbox[2] and bbox[1] < pt[1] < bbox[1] + bbox[3]

def get_area(bbox):
    _, _, w, h = bbox
    return w * h