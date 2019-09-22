import cv2
from util.logger import log_info
import base64
import pathlib
import uuid
import os


def take_screenshot(frame):
    screenshots_directory = 'data/screenshots'
    pathlib.Path(screenshots_directory).mkdir(parents=True, exist_ok=True)
    screenshot_path = os.path.join(screenshots_directory, 'img_' + uuid.uuid4().hex + '.jpg')
    cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    log_info('Screenshot captured.', {
        'cat': 'SCREENSHOT_CAPTURE',
        'path': screenshot_path,
    })

def get_base64_image(image):
    try:
        _, image_buffer = cv2.imencode('.jpg', image)
        image_str = base64.b64encode(image_buffer).decode('utf-8')
        return 'data:image/jpeg;base64, {0}'.format(image_str)
    except:
        return None