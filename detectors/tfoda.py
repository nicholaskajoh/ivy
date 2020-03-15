'''
Perform detection using models created with the Tensorflow Object Detection API.
https://github.com/tensorflow/models/tree/master/research/object_detection
'''

# pylint: disable=missing-function-docstring,invalid-name

import numpy as np
import tensorflow as tf
import settings


with open(settings.TFODA_CLASSES_PATH, 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(settings.TFODA_CLASSES_OF_INTEREST_PATH, 'r') as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])

def scale_box_coords(box, img_w, img_h):
    # The box is in normalised order [ymin, xmin, ymax, xmax]
    return [
        box[1] * img_w, # xmin
        box[0] * img_h, # ymin
        (box[3] - box[1]) * img_w, # width
        (box[2] - box[0]) * img_h, # height
    ]

confidence_threshold = float(settings.TFODA_CONFIDENCE_THRESHOLD)
model_dir = settings.TFODA_MODEL_DIR

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
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    _classes = []
    _confidences = []
    _bounding_boxes = []

    for i, class_id in enumerate(output_dict['detection_classes']):
        confidence = output_dict['detection_scores'][i]
        _class = CLASSES[class_id]
        if confidence > confidence_threshold and _class in CLASSES_OF_INTEREST:
            _box = scale_box_coords(output_dict['detection_boxes'][i], img_w, img_h)
            _classes.append(_class)
            _confidences.append(confidence)
            _bounding_boxes.append(_box)

    return _bounding_boxes, _classes, _confidences
