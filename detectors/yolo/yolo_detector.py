import cv2
import numpy as np
import os
from configparser import ConfigParser

class VehicleCountingConfigNotFound(Exception):
    pass

class YoloConfigSectionNotFoundError(Exception):
    pass

class WeightsConfigOptionNotFoundError(Exception):
    pass

class WeightsConfigOptionFileNotFoundError(Exception):
    pass

class WeightsAndConfigNotFoundError(Exception):
    pass

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
VC_CONFIG_YOLO_SECTION = 'yolo'
VC_CONFIG_YOLO_WEIGHTS_OPTION = 'weights'
VC_CONFIG_YOLO_CONFIG_OPTION = 'yolov3.cfg'
DEFAULT_WEIGHTS_FILE = 'yolov3.weights'
def get_bounding_boxes(image):
    # get object classes
    classes = None
    with open(os.path.join(__location__, 'classes.txt'), 'r') as classes_file:
        classes = [line.strip() for line in classes_file.readlines()]
    classes_of_interest = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
    
    # create a YOLO v3 DNN model using pre-trained weights
    net = None
    weights_file = os.path.join(__location__, DEFAULT_WEIGHTS_FILE)
    yolo_cfg_file  = os.path.join(__location__, 'yolov3.cfg')

    is_weights_exists = os.path.isfile(weights_file)
    is_cfg_exists = os.path.isfile(yolo_cfg_file)
    
    config_parser = ConfigParser()
    # if either the weight or cfg is absent 
    if not is_weights_exists:  
        error_message = '\nUnable to locate yolo.weights in default location\n'
        weight_found = False              
        
        # check for a config file in the current working directory
        vc_config = os.path.join(os.getcwd(), 'vc.cfg')

        #if vehicle counting config  file exists
        if(os.path.isfile(vc_config)):
            config_parser.read(vc_config)
            #check for yolo weights config secton
            if(config_parser.has_section(VC_CONFIG_YOLO_SECTION)):
                #check for weights options in section
                if(config_parser.has_option(VC_CONFIG_YOLO_SECTION, VC_CONFIG_YOLO_WEIGHTS_OPTION)):
                    #check if the file in weights option exists
                    if(os.path.isfile(config_parser.get(VC_CONFIG_YOLO_SECTION,VC_CONFIG_YOLO_WEIGHTS_OPTION))):
                        weights_file = config_parser.get(VC_CONFIG_YOLO_SECTION,VC_CONFIG_YOLO_WEIGHTS_OPTION)
                        weight_found = True
                    else:
                        error_message += 'The file specified in the yolo weight option of the Vehicle-Counting config does not exist\n'
                        raise WeightsConfigOptionFileNotFoundError(error_message)
                else:
                    error_message += 'A weights option was not found in yolo section of the Vehicle-Counting config file\n'  
                    raise YoloConfigSectionNotFoundError(error_message)
            else:
                error_message += 'A yolo section was not found in the Vehicle-Counting config file\n'
                raise YoloConfigSectionNotFoundError(error_message)
        else: # config file 
            error_message+='Vehicle Counting config file not found\n'
            raise VehicleCountingConfigNotFound(error_message)

    # make sure yolo confi file exists
    if(not is_cfg_exists):
        exit("yolo config file not found")

    try:
        net = cv2.dnn.readNet(weights_file, yolo_cfg_file)
    except:
        exit("\n\nUnable to create yolo DDN model")

    # create image blob
    scale = 0.00392
    image_blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # detect objects
    net.setInput(image_blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in classes_of_interest:
                width = image.shape[1]
                height = image.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    _bounding_boxes = []
    for i in indices:
        i = i[0]
        _bounding_boxes.append(boxes[i])

    return _bounding_boxes