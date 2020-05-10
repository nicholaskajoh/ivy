'''
Perform detection using models created with FAIR's Detectron2 framework.
https://github.com/facebookresearch/detectron2
'''

# pylint: disable=import-error,no-name-in-module,invalid-name,broad-except

import torch
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import settings
from util.logger import get_logger


setup_logger()
logger = get_logger()

with open(settings.DETECTRON2_CLASSES_PATH, 'r') as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(settings.DETECTRON2_CLASSES_OF_INTEREST_PATH, 'r') as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])

# initialize model with weights and config
cfg = get_cfg()
cfg.merge_from_file(settings.DETECTRON2_CONFIG_PATH)
cfg.MODEL.WEIGHTS = settings.DETECTRON2_WEIGHTS_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(settings.DETECTRON2_NUM_CLASSES)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(settings.DETECTRON2_CONFIDENCE_THRESHOLD)
cfg.DATALOADER.NUM_WORKERS = 2

if torch.cuda.is_available():
    logger.debug('GPU available, using GPU')
    cfg.MODEL.DEVICE = 'cuda'
else:
    logger.debug('No GPU available, using CPU')
    cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

def convert_box_to_array(box):
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
    for val in box:
        for ind_val in val:
            res.append(float(ind_val))

    # Convert x2, y2 into w, h
    res[2] = res[2] - res[0]
    res[3] = res[3] - res[1]
    return res

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of objects detected,
    their classes and the confidences of the detections made.
    '''
    try:
        outputs = predictor(image)
    except Exception as error:
        logger.error(error)

    _classes = []
    _confidences = []
    _bounding_boxes = []

    for i, pred in enumerate(outputs["instances"].pred_classes):
        class_id = int(pred)
        _class = CLASSES[class_id]

        if _class in CLASSES_OF_INTEREST:
            _classes.append(_class)

            confidence = float(outputs['instances'].scores[i])
            _confidences.append(confidence)

            _box = outputs['instances'].pred_boxes[i]
            box_array = convert_box_to_array(_box)
            _bounding_boxes.append(box_array)

    return _bounding_boxes, _classes, _confidences
