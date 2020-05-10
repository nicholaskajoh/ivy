'''
Utilities for configuring and debugging the object count process.
'''

import cv2

from .logger import get_logger
import settings


logger = get_logger()

def mouse_callback(event, x, y, flags, param):
    '''
    Handler for mouse events in the debug window.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_pixel_position(x, y, param['frame_width'], param['frame_height'])

def capture_pixel_position(window_x, window_y, frame_w, frame_h):
    '''
    Capture the position of a pixel in a video frame.
    '''
    debug_window_size = settings.DEBUG_WINDOW_SIZE
    x = round((frame_w / debug_window_size[0]) * window_x)
    y = round((frame_h / debug_window_size[1]) * window_y)
    logger.info('Pixel position captured.', extra={'meta': {'label': 'PIXEL_POSITION', 'position': (x, y)}})
