"""
Perform detection using models created with the Tensorflow Object Detection API.
https://github.com/tensorflow/models/tree/master/research/object_detection
This class uses tensorflow to load and run the models, as opposed to tfoda.py
which uses OpenCV to load and run the TF models.
"""
import numpy as np
import os
from pathlib import Path
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

def scale_box_coords(box, img_w, img_h):
    res = []

    # The box is in normalised order [ymin, xmin, ymax, xmax]
    res.append(box[1] * img_w) #xmin
    res.append(box[0] * img_h) #ymin
    res.append((box[3] - box[1]) * img_w) #width
    res.append((box[2] - box[0]) * img_h) #height

    return res

with open(os.getenv('TFODA_CLASSES_PATH'), 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))

confidence_threshold = float(os.getenv('TFODA_CONFIDENCE_THRESHOLD'))
model_dir = os.getenv('TFODA_WEIGHTS_PATH')

model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']

def get_bounding_boxes(image):
    image = image[:, :, ::-1]
    img_h, img_w, _ = image.shape

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Run inference
    output_dict = model(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    _classes = []
    _confidences = []
    _bounding_boxes = []

    for i, pred in enumerate(output_dict['detection_classes']):
        score = round(float(output_dict['detection_scores'][i]), 3)
        if score < confidence_threshold:
            continue
        coco_class = CLASSES[int(pred)]
        this_box = scale_box_coords(output_dict['detection_boxes'][i], img_w, img_h)
        _classes.append(coco_class)
        _confidences.append(score)
        _bounding_boxes.append(this_box)

    return _bounding_boxes, _classes, _confidences