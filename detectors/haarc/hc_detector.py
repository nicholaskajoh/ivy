import cv2


def get_bounding_boxes(frame):
    fullbody_cascade = cv2.CascadeClassifier('./HaarCascades/car.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = fullbody_cascade.detectMultiScale(gray)
    return bounding_boxes