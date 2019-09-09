import sys
sys.path.append('..')

import cv2
import base64


def get_base64_image(image):
    _, image_buffer = cv2.imencode('.jpg', image)
    image_str = base64.b64encode(image_buffer).decode('utf-8')
    return 'data:image/jpeg;base64, {0}'.format(image_str)