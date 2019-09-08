import cv2
import os
import sys
import pathlib
import json
import uuid
from datetime import datetime


def send_to_stdout(level, message, data):
    events_to_not_log = ['TRACKER_UPDATE']
    if data['event'] not in events_to_not_log:
        print('[{0}]'.format(datetime.now()), level + ':', message, json.dumps(data))

def send_to_log_file():
    pass

def send_to_redis_pubsub():
    pass

def log_error(message, data):
    send_to_stdout('ERROR', message, data)
    sys.exit(0)

def log_info(message, data):
    send_to_stdout('INFO', message, data)

def log_debug(message, data):
    send_to_stdout('DEBUG', message, data)

def take_screenshot(frame):
    screenshots_directory = 'data/screenshots'
    pathlib.Path(screenshots_directory).mkdir(parents=True, exist_ok=True)
    screenshot_path = os.path.join(screenshots_directory, 'img_' + uuid.uuid4().hex + '.jpg')
    cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    log_info('Screenshot captured.', {
        'event': 'SCREENSHOT_CAPTURE',
        'path': screenshot_path,
    })