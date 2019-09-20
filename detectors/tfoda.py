"""
Perform detection using models created with the Tensorflow Object Detection API.
https://github.com/tensorflow/models/tree/master/research/object_detection
"""

import cv2
import numpy as np
import os


with open(os.getenv('TFODA_CLASSES_PATH'), 'r') as classes_file:
    classes = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(os.getenv('TFODA_CLASSES_OF_INTEREST_PATH'), 'r') as coi_file:
    classes_of_interest = tuple([line.strip() for line in coi_file.readlines()])

def get_bounding_boxes(image):
    # create model using weights and config
    net = cv2.dnn.readNetFromTensorflow(os.getenv('TFODA_WEIGHTS_PATH'), os.getenv('TFODA_CONFIG_PATH'))

    # create image blob
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    # detect objects
    net.setInput(blob)
    detections = net.forward()

    # get bounding boxes
    confidence_threshold = float(os.getenv('TFODA_CONFIDENCE_THRESHOLD'))
    boxes = []
    _classes = []
    _confidences = []
    rows, cols, _ = image.shape
    for detection in detections[0, 0]:
        confidence = float(detection[2])
        class_id = int(detection[1])
        if confidence > confidence_threshold and class_id in classes and classes[class_id] in classes_of_interest:
            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)

            x, y, w, h = left, top, right - left, bottom - top
            boxes.append([x, y, w, h])
            _classes.append(classes[class_id])
            _confidences.append(confidence)

    # remove overlapping bounding boxes
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, _confidences, confidence_threshold, nms_threshold)

    _bounding_boxes = []
    for i in indices:
        i = i[0]
        _bounding_boxes.append(boxes[i])

    return _bounding_boxes, _classes, _confidences