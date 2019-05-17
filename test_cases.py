import os
import unittest
import detectors.yolo.yolo_detector  as yolo
vc_config = os.path.join(os.getcwd(), 'vc.cfg')

class YoloTest(unittest.TestCase):
    
    
    def test_no_weights_no_config_file_should_throw_error(self):
        """
        Test that a value error is thrown when yolo weights file is found in the default location
        and a vehicle counting config file is not found
        """
        os.remove(vc_config)
        with self.assertRaises(yolo.VehicleCountingConfigNotFound):
            yolo.get_bounding_boxes(None)

    def test_no_default_no_yolo_section_should_throw_error(self):
        """
        Test that an error is thrown when yolo weights are not found in the default location
        and a yolo section does not exist in the config file
        """
        f= open(vc_config,"w+")
        f.close()
        with self.assertRaises(yolo.YoloConfigSectionNotFoundError):
            yolo.get_bounding_boxes(None)

    def test_no_default_no_weights_option_should_throw_error(self):
        """
        Test that a value error is thrown when yolo weights are not found in the default location
        and a yolo weight option is not found in the config option
        """
        f= open(vc_config,"w+")        
        f.write("["+yolo.VC_CONFIG_YOLO_SECTION+"]")
        f.close()
        
        with self.assertRaises(yolo.YoloConfigSectionNotFoundError):
            yolo.get_bounding_boxes(None)

if __name__ == '__main__':
    #yolo.get_bounding_boxes(None)
    unittest.main()