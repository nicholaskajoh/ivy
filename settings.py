'''
App settings.
'''

import os
import ast


ENVS_READY = True

# Specify if video capture is from a camera
try:
    IS_CAM = ast.literal_eval(os.getenv('IS_CAM', 'False'))
except ValueError:
    print('Invalid value for IS_CAM. It should be either True or False.')
    ENVS_READY = False

# Absolute/relative path to video or camera input
# E.g "./data/videos/sample_traffic_scene.mp4" or 1
if os.getenv('VIDEO'):
    VIDEO = int(os.getenv('VIDEO')) if IS_CAM else os.getenv('VIDEO')
else:
    print('Path to video or camera input not set.')
    ENVS_READY = False

# Specify a detection Region of Interest (ROI)
# i.e a set of vertices that represent the area (polygon) where you want detections to be made
# E.g [(750, 405), (1094, 398), (1569, 1028), (501, 1028)]
# Default: [(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)] (i.e the whole video frame)
try:
    USE_DROI = ast.literal_eval(os.getenv('USE_DROI', 'False'))
except ValueError:
    print('Invalid value for USE_DROI. It should be either True or False.')
    ENVS_READY = False

if USE_DROI:
    try:
        DROI = ast.literal_eval(os.getenv('DROI'))
    except ValueError:
        print('Invalid value for DROI. It should be a list of coordinates (2-tuples).')
        ENVS_READY = False

# Display/overlay the detection ROI on the video
try:
    SHOW_DROI = ast.literal_eval(os.getenv('SHOW_DROI', 'False'))
except ValueError:
    print('Invalid value for SHOW_DROI. It should be either True or False.')
    ENVS_READY = False

# Display cumulative counts on the video
try:
    SHOW_COUNTS = ast.literal_eval(os.getenv('SHOW_COUNTS', 'True'))
except ValueError:
    print('Invalid value for SHOW_COUNTS. It should be either True or False.')
    ENVS_READY = False

# Maximum consecutive detection failures i.e number of detection failures
# before it's concluded that an object is no longer in the frame
try:
    MCDF = int(os.getenv('MCDF', '2'))
except ValueError:
    print('Invalid value for MCDF. It should be a positive integer.')
    ENVS_READY = False

# Maximum consecutive tracking failures i.e number of tracking failures
# before the tracker concludes the tracked object has left the frame
try:
    MCTF = int(os.getenv('MCTF', '3'))
except ValueError:
    print('Invalid value for MCTF. It should be a positive integer.')
    ENVS_READY = False

# Detection interval i.e number of frames before detection is carried out again
# (in order to find new vehicles and update the trackers of old ones)
try:
    DI = int(os.getenv('DI', '10'))
except ValueError:
    print('Invalid value for DI. It should be a positive integer.')
    ENVS_READY = False

# Model/algorithm to use for vehicle detection (options: yolo, tfoda, detectron2, haarcascade)
DETECTOR = os.getenv('DETECTOR', 'yolo')

# Algorithm to use for vehicle tracking (options: kcf, csrt)
TRACKER = os.getenv('TRACKER', 'kcf')

# Record vehicle counting as video
try:
    RECORD = ast.literal_eval(os.getenv('RECORD', 'False'))
except ValueError:
    print('Invalid value for RECORD. It should be either True or False.')
    ENVS_READY = False

# Set path where recorded video will be stored
if RECORD:
    if os.getenv('OUTPUT_VIDEO_PATH'):
        OUTPUT_VIDEO_PATH = os.getenv('OUTPUT_VIDEO_PATH')
    else:
        print('Output video path not set.')
        ENVS_READY = False

# Run VCS without UI display
try:
    HEADLESS = ast.literal_eval(os.getenv('HEADLESS', 'False'))
except ValueError:
    print('Invalid value for HEADLESS. It should be either True or False.')
    ENVS_READY = False

# Specify one or more counting lines
# A counting line is represented by a label and line segment
# E.g {'label': 'A', 'line': [(667, 713), (888, 713)]}
if os.getenv('COUNTING_LINES'):
    COUNTING_LINES = ast.literal_eval(os.getenv('COUNTING_LINES'))
else:
    print('Invalid value for COUNTING_LINES. It should be a list of lines.')
    ENVS_READY = False

# Configs for Haar Cascade detector
if DETECTOR == 'haarcascade':
    if os.getenv('HAAR_CASCADE_PATH'):
        HAAR_CASCADE_PATH = os.getenv('HAAR_CASCADE_PATH')
    else:
        print('HAAR_CASCADE_PATH not set.')
        ENVS_READY = False

# Configs for TFODA (Tensorflow Object Detection API) detector
if DETECTOR == 'tfoda' or DETECTOR == 'tfoda_new':
    if os.getenv('TFODA_WEIGHTS_PATH') and \
            os.getenv('TFODA_CONFIG_PATH') and \
            os.getenv('TFODA_MODEL_DIR') and \
            os.getenv('TFODA_CLASSES_PATH') and \
            os.getenv('TFODA_CLASSES_OF_INTEREST_PATH') and \
            os.getenv('TFODA_CONFIDENCE_THRESHOLD'):
        TFODA_WEIGHTS_PATH = os.getenv('TFODA_WEIGHTS_PATH')
        TFODA_CONFIG_PATH = os.getenv('TFODA_CONFIG_PATH')
        TFODA_MODEL_DIR = os.getenv('TFODA_MODEL_DIR')
        TFODA_CLASSES_PATH = os.getenv('TFODA_CLASSES_PATH')
        TFODA_CLASSES_OF_INTEREST_PATH = os.getenv('TFODA_CLASSES_OF_INTEREST_PATH')
        TFODA_CONFIDENCE_THRESHOLD = float(os.getenv('TFODA_CONFIDENCE_THRESHOLD'))
    else:
        print('TFODA_WEIGHTS_PATH, TFODA_CONFIG_PATH, TFODA_MODEL_DIR, TFODA_CLASSES_PATH, ' +
              'TFODA_CLASSES_OF_INTEREST_PATH and/or TFODA_CONFIDENCE_THRESHOLD not set or invalid.')
        ENVS_READY = False

# Configs for YOLO (You Only Look Once) detector
if DETECTOR == 'yolo':
    if os.getenv('YOLO_WEIGHTS_PATH') and \
            os.getenv('YOLO_CONFIG_PATH') and \
            os.getenv('YOLO_CLASSES_PATH') and \
            os.getenv('YOLO_CLASSES_OF_INTEREST_PATH') and \
            os.getenv('YOLO_DATA_PATH') and \
            os.getenv('YOLO_CONFIDENCE_THRESHOLD'):
        YOLO_WEIGHTS_PATH = os.getenv('YOLO_WEIGHTS_PATH')
        YOLO_CONFIG_PATH = os.getenv('YOLO_CONFIG_PATH')
        YOLO_CLASSES_PATH = os.getenv('YOLO_CLASSES_PATH')
        YOLO_CLASSES_OF_INTEREST_PATH = os.getenv('YOLO_CLASSES_OF_INTEREST_PATH')
        YOLO_DATA_PATH = os.getenv('YOLO_DATA_PATH')
        YOLO_CONFIDENCE_THRESHOLD = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD'))
    else:
        print('YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH, YOLO_CLASSES_PATH, YOLO_CLASSES_OF_INTEREST_PATH, ' +
              'YOLO_DATA_PATH and/or YOLO_CONFIDENCE_THRESHOLD not set or invalid.')
        ENVS_READY = False

# Configs for Detectron2 detector
if DETECTOR == 'detectron2':
    if os.getenv('DETECTRON2_CONFIDENCE_THRESHOLD') and \
            os.getenv('DETECTRON2_CONFIG_PATH') and \
            os.getenv('DETECTRON2_WEIGHTS_PATH') and \
            os.getenv('DETECTRON2_NUM_CLASSES') and \
            os.getenv('DETECTRON2_CLASSES_PATH') and \
            os.getenv('DETECTRON2_CLASSES_OF_INTEREST_PATH'):
        DETECTRON2_CONFIDENCE_THRESHOLD = float(os.getenv('DETECTRON2_CONFIDENCE_THRESHOLD'))
        DETECTRON2_CONFIG_PATH = os.getenv('DETECTRON2_CONFIG_PATH')
        DETECTRON2_WEIGHTS_PATH = os.getenv('DETECTRON2_WEIGHTS_PATH')
        DETECTRON2_NUM_CLASSES = int(os.getenv('DETECTRON2_NUM_CLASSES'))
        DETECTRON2_CLASSES_PATH = os.getenv('DETECTRON2_CLASSES_PATH')
        DETECTRON2_CLASSES_OF_INTEREST_PATH = os.getenv('DETECTRON2_CLASSES_OF_INTEREST_PATH')
    else:
        print('DETECTRON2_CONFIDENCE_THRESHOLD, DETECTRON2_CONFIG_PATH, DETECTRON2_WEIGHTS_PATH, DETECTRON2_NUM_CLASSES, ' +
              'DETECTRON2_CLASSES_PATH and/or DETECTRON2_CLASSES_OF_INTEREST_PATH not set or invalid.')
        ENVS_READY = False

# Log destinations
try:
    ENABLE_CONSOLE_LOGGER = ast.literal_eval(os.getenv('ENABLE_CONSOLE_LOGGER', 'True'))
    ENABLE_FILE_LOGGER = ast.literal_eval(os.getenv('ENABLE_FILE_LOGGER', 'True'))
    ENABLE_LOGSTASH_LOGGER = ast.literal_eval(os.getenv('ENABLE_LOGSTASH_LOGGER', 'False'))
except ValueError:
    print('Invalid value for ENABLE_CONSOLE_LOGGER, ENABLE_FILE_LOGGER ' +
          'and/or ENABLE_LOGSTASH_LOGGER. They should be either True or False.')
    ENVS_READY = False

# Absolute/relative path to log files directory
if ENABLE_FILE_LOGGER:
    LOG_FILES_DIRECTORY = os.getenv('LOG_FILES_DIRECTORY', './data/logs/')

# Size of window used to view the vehicle counting process
try:
    DEBUG_WINDOW_SIZE = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE', '(858, 480)'))
except ValueError:
    print('Invalid value for DEBUG_WINDOW_SIZE. It should be a 2-tuple: (width, height).')
    ENVS_READY = False


if not ENVS_READY:
    raise Exception('One or more environment variables are either invalid or not set. ' +
                    'Please ensure all variables are properly set.')
