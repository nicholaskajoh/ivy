'''
Bounding box utility functions.
'''

def get_centroid(bbox):
    '''
    Calculates the center point of a bounding box.
    '''
    x, y, w, h = bbox
    return (round((x + x + w) / 2), round((y + y + h) / 2))

def box_contains_point(bbox, point):
    '''
    Checks if a given point is within a bounding box.
    '''
    return bbox[0] < point[0] < bbox[0] + bbox[2] and bbox[1] < point[1] < bbox[1] + bbox[3]

def get_area(bbox):
    '''
    Calculates the area of a bounding box.
    '''
    _, _, w, h = bbox
    return w * h

def get_overlap(bbox1, bbox2):
    '''
    Calculates the degree of overlap of two bounding boxes.
    This can be any value from 0 to 1 where 0 means no overlap and 1 means complete overlap.
    The degree of overlap is the ratio of the area of overlap of two boxes and the area of the smaller box.
    '''
    bbox1_x1 = bbox1[0]
    bbox1_y1 = bbox1[1]
    bbox1_x2 = bbox1[0] + bbox1[2]
    bbox1_y2 = bbox1[1] + bbox1[3]

    bbox2_x1 = bbox2[0]
    bbox2_y1 = bbox2[1]
    bbox2_x2 = bbox2[0] + bbox2[2]
    bbox2_y2 = bbox2[1] + bbox2[3]

    overlap_x1 = max(bbox1_x1, bbox2_x1)
    overlap_y1 = max(bbox1_y1, bbox2_y1)
    overlap_x2 = min(bbox1_x2, bbox2_x2)
    overlap_y2 = min(bbox1_y2, bbox2_y2)

    overlap_width = overlap_x2 - overlap_x1
    overlap_height = overlap_y2 - overlap_y1

    if overlap_width < 0 or overlap_height < 0:
        return 0.0

    overlap_area = overlap_width * overlap_height

    bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
    bbox2_area = (bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1)
    smaller_area = bbox1_area if bbox1_area < bbox2_area else bbox2_area

    epsilon = 1e-5 # small value to prevent division by zero
    overlap = overlap_area / (smaller_area + epsilon)
    return overlap

def get_box_image(frame, bbox):
    '''
    Fetches the image of the area covered by a bounding box.
    '''
    x, y, w, h = list(map(int, bbox))
    return frame[y - 10:y + h + 10, x - 10:x + w + 10] # allowance of 10 pixels on every side
