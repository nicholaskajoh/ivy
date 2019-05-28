import sys
import os

test_dir = os.path.abspath(os.path.join(__file__, os.pardir))
vc_dir = os.path.dirname(test_dir)
sys.path.append(vc_dir)

import unittest
from detectors.yolo import yolo_detector  as yolo
import cv2
vc_config = os.path.join(os.getcwd(), 'vc.cfg')

class YoloTest(unittest.TestCase):
    
    def test_no_default_weights_option_file_exist(self):
        """
        Test that an error is thrown when yolo weights file is not in the default location (detector/yolo),
        a vehicle counting config file (vc.cfg) is found in the CWD,
        a yolo configuration section exist in the config file,
        a yolo weights option exists in the config
        and the file in yolo weights option of vehicle counting config file (vc.cfg) exists
        """
        f = open(vc_config,"w+")       
        f.write("["+yolo.VC_CONFIG_YOLO_SECTION+"]\n")
        f.write(yolo.VC_CONFIG_YOLO_WEIGHTS_OPTION+" = "+os.path.join(os.getcwd(), 'yolo.weights')+"\n")        
        f.close()

        f = open(os.path.join(os.getcwd(), 'yolo.weights'), "w+")
        f.write("")    
        f.close()

        # because the file is not a valid weights file and a cv2 error (because of the invalid model) was caught in the
        # yolo detector 
        with self.assertRaises(cv2.error):
            yolo.get_bounding_boxes(None)

    def test_no_default_weights_option_file_not_exist_should_throw_WeightsConfigOptionFileNotFoundError(self):
        """
        Test that an error is thrown when yolo weights file is not in the default location (detector/yolo),
        a vehicle counting config file (vc.cfg) is found in the CWD,
        a yolo configuration section exist in the config file,
        a yolo weights option exists in the config
        but the file in yolo weights option of vehicle counting config file (vc.cfg) does not exist
        """
        os.remove(os.path.join(os.getcwd(), 'yolo.weights'))
        f= open(vc_config,"w+")       
        f.write("["+yolo.VC_CONFIG_YOLO_SECTION+"]\n")
        f.write(yolo.VC_CONFIG_YOLO_WEIGHTS_OPTION+" = yolo.weights\n")
        f.close()
        
        with self.assertRaises(yolo.WeightsConfigOptionFileNotFoundError):
            yolo.get_bounding_boxes(None)
 
    def test_no_default_no_weights_option_should_throw_YoloConfigSectionNotFoundError(self):
        """
        Test that an error is thrown when yolo weights file is not in the default location (detector/yolo),
        a vehicle counting config file (vc.cfg) is found in the CWD,
        a yolo configuration section exist in the config file
        but a yolo weight option is not found in the config option
        """
        f= open(vc_config,"w+")        
        f.write("["+yolo.VC_CONFIG_YOLO_SECTION+"]\n")
        f.close()
        
        with self.assertRaises(yolo.YoloConfigSectionNotFoundError):
            yolo.get_bounding_boxes(None)

    def test_no_default_no_yolo_section_should_throw_error(self):
        """
        Test that an error is thrown when yolo weights file is not in the default location (detector/yolo),
        a vehicle counting config file (vc.cfg) is found in the CWD
        but a yolo configuration section does not exist in the config file
        """
        f= open(vc_config,"w+")
        f.close()
        with self.assertRaises(yolo.YoloConfigSectionNotFoundError):
            yolo.get_bounding_boxes(None)

    def test_no_weights_no_config_file_should_throw_error(self):
        """
        Test that a value error is thrown when yolo weights file is not in the default location 'detector/yolo'
        and a vehicle counting config file (vc.cfg) is not found in the CWD
        """
        os.remove(vc_config)
        with self.assertRaises(yolo.VehicleCountingConfigNotFound):
            yolo.get_bounding_boxes(None)

if __name__ == '__main__':
    print('*'*100)
    print("Testing")
    print('*'*100)
    print(sys.path)
    unittest.main()