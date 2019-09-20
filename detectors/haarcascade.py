import cv2
import os


def get_bounding_boxes(frame):
    car_cascade = cv2.CascadeClassifier(os.getenv('HAAR_CASCADE_PATH'))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = car_cascade.detectMultiScale(gray)
    return bounding_boxes, None, None