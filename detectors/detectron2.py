'''
Performs detection using models created with FAIRs Detectron2 Framework.
https://github.com/facebookresearch/detectron2
'''

import ast
import os
import cv2
import torch
import numpy as np
import random

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

setup_logger()

with open(os.getenv('DETECTRON2_CLASSES_PATH'), 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(os.getenv('DETECTRON2_CLASSES_OF_INTEREST_PATH'), 'r') as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])

# initialize model with weights and config
cfg = get_cfg()
cfg.merge_from_file(os.getenv('DETECTRON2_CONFIG_PATH'))
cfg.MODEL.WEIGHTS = os.getenv('DETECTRON2_WEIGHTS_PATH')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(os.getenv('DETECTRON2_NUM_CLASSES'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(os.getenv('DETECTRON2_CONFIDENCE_THRESHOLD'))
cfg.DATALOADER.NUM_WORKERS = 2

from util.logger import get_logger
logger = get_logger()

if not torch.cuda.is_available():
    logger.debug('No GPU available, using CPU')
    cfg.MODEL.DEVICE = 'cpu'
else:
    logger.debug('GPU Available, using GPU')
    cfg.MODEL.DEVICE = 'cuda'

predictor = DefaultPredictor(cfg)

def convert_box_to_array(this_box):
    '''
    Detectron2 returns results as a Boxes class 
    (https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Boxes).
    These have a lot of useful extra methods (computing area etc) however can't be easily
    serialized, so it's a pain to transfer these predictions over the internet (e.g., if
    this is being used in the cloud, and wanting to send predictions back) so this method
    converts it into a standard array format. Detectron2 also returns results in (x1, y1, x2, y2) 
    format, so this method converts it into (x1, y1, w, h).
    '''
    res = []
    for val in this_box:
        for ind_val in val:
            res.append(float(ind_val))

    # Convert x2, y2 into w, h
    res[2] = res[2] - res[0]
    res[3] = res[3] - res[1]
    return res

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of vehicles detected,
    their classes and the confidences of the detections made.
    '''
    try:
      outputs = predictor(image)
    except Exception as e:
      print(e)

    _classes = []
    _confidences = []
    _bounding_boxes = []

    for i, pred in enumerate(outputs["instances"].pred_classes):
        coco_class = int(pred)
        class_string = CLASSES[coco_class]
        _classes.append(class_string)

        score = round(float(outputs['instances'].scores[i]), 3)
        _confidences.append(score)

        this_box = outputs['instances'].pred_boxes[i]
        box_vals = convert_box_to_array(this_box)
        _bounding_boxes.append(box_vals)

    return _bounding_boxes, _classes, _confidences
