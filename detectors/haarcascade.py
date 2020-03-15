'''
Perform detection using Haar feature-based cascade classifiers.
'''

# pylint: disable=missing-function-docstring

import cv2
import settings


def get_bounding_boxes(frame):
    object_cascade = cv2.CascadeClassifier(settings.HAAR_CASCADE_PATH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = object_cascade.detectMultiScale(gray)
    return bounding_boxes, None, None
