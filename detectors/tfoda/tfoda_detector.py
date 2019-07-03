"""
Perform detection using models created with the Tensorflow Object Detection API.
https://github.com/tensorflow/models/tree/master/research/object_detection
"""

import cv2
import numpy as np
import os
from dotenv import load_dotenv


load_dotenv()
with open(os.getenv('CLASSES_PATH'), 'r') as classes_file:
    classes = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(os.getenv('CLASSES_OF_INTEREST_PATH'), 'r') as coi_file:
    classes_of_interest = tuple([line.strip() for line in coi_file.readlines()])

def get_bounding_boxes(image):
    # create model using weights and config
    net = cv2.dnn.readNetFromTensorflow(os.getenv('WEIGHTS_PATH'), os.getenv('CONFIG_PATH'))

    # create image blob
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    # detect objects
    net.setInput(blob)
    detections = net.forward()

    # get bounding boxes
    bounding_boxes = []
    rows, cols, _ = image.shape
    for detection in detections[0, 0]:
        confidence = float(detection[2])
        class_id = int(detection[1])
        if confidence > float(os.getenv('CONFIDENCE_THRESHOLD')) and class_id in classes and classes[class_id] in classes_of_interest:
            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)

            x, y, w, h = left, top, right - left, bottom - top
            bounding_boxes.append([x, y, w, h])

    return bounding_boxes