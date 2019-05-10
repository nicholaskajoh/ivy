import cv2
import numpy as np
import os


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_bounding_boxes(image):
    # get object classes
    classes = {0: 'background',
                1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    classes_of_interest = ('bicycle', 'car', 'motorbike', 'bus')

    # create an SSD model using pre-trained weights
    # from https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view
    prototxt = os.path.join(__location__, 'deploy.prototxt')
    weights = os.path.join(__location__, 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    net = cv2.dnn.readNetFromCaffe(prototxt, weights)

    # create image blob
    resized_image = cv2.resize(image, (300, 300)) # 300 x 300
    blob = cv2.dnn.blobFromImage(resized_image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # detect objects
    net.setInput(blob)
    detections = net.forward()

    # get bounding boxes
    bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if confidence > 0.2 and class_id in classes and classes[class_id] in classes_of_interest:
            x_bottom_left = int(detections[0, 0, i, 3] * 300)
            y_bottom_left = int(detections[0, 0, i, 4] * 300)
            x_top_right = int(detections[0, 0, i, 5] * 300)
            y_top_right = int(detections[0, 0, i, 6] * 300)

            height_factor = image.shape[0] / 300.0
            width_factor = image.shape[1] / 300.0

            # scale detection to orginal image size
            x_bottom_left = int(width_factor * x_bottom_left) 
            y_bottom_left = int(height_factor * y_bottom_left)
            x_top_right = int(width_factor * x_top_right)
            y_top_right = int(height_factor * y_top_right)

            x, y, w, h = x_bottom_left, y_bottom_left, x_top_right - x_bottom_left, y_top_right - y_bottom_left
            bounding_boxes.append([x, y, w, h])

    return bounding_boxes