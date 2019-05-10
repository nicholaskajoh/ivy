import sys
sys.path.append('..')

import cv2
import numpy as np
from blobs.blob3 import Blob


class Camshift():
    def __init__(self, _bounding_box, _roi_hist):
        self.bounding_box = _bounding_box
        self.roi_hist = _roi_hist

    def update(self, frame):
        term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, _ = cv2.CamShift(mask, tuple(int(v) for v in self.bounding_box), term_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        bounding_box = cv2.boundingRect(pts)
        x, y, w, h = [int(v) for v in bounding_box]

        _roi = frame[y:y + h, x:x + w]
        _hsv_roi = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV)
        _roi_hist = cv2.calcHist([_hsv_roi], [0], None, [180], [0, 180])
        correlation = cv2.compareHist(_roi_hist, self.roi_hist, cv2.HISTCMP_CORREL)
        
        success = False
        if (correlation >= 0.5):
            success = True

        return success, bounding_box


def camshift_create(bounding_box, frame):
    x, y, w, h = [int(v) for v in bounding_box]
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    tracker = Camshift(bounding_box, roi_hist)
    return Blob((x, y, w, h), roi_hist, tracker)