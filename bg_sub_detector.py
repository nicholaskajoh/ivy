import cv2
import math
from blob2 import Blob


fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

def get_bounding_boxes(image):
    fgmask = fgbg.apply(image)

    # find and draw contours on image
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fgmask, contours, -1, (255, 255, 255), -1)

    # get convex hulls from contours
    convex_hulls = []
    for i in range(len(contours)):
        convex_hulls.append(cv2.convexHull(contours[i], False))

    bboxes = []
    # filter out bounding boxes that are not sizeable enough to be a vehicle
    # TODO: more work needed to fine tune this filter (quite a number of false positives)
    for convex_hull in convex_hulls:
        x, y, w, h = cv2.boundingRect(convex_hull)
        bbox = (x, y, w, h)
        bbox_area = w * h
        aspect_ratio = float(w) / float(h)
        diagonal_size = math.sqrt(math.pow(w, 2) + math.pow(h, 2))

        if bbox_area > 200 and \
                aspect_ratio > 0.2 and \
                aspect_ratio < 4.0 and \
                w > 30 and \
                h > 30 and \
                diagonal_size > 100.0:
            bboxes.append(bbox)

    return bboxes