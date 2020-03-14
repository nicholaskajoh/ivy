'''
Test debugger util.
'''

import logging
import settings
from util.debugger import capture_pixel_position


def test_capture_pixel_position(caplog):
    # pylint: disable=missing-function-docstring
    settings.DEBUG_WINDOW_SIZE = (1280, 720)
    caplog.set_level(logging.INFO)
    capture_pixel_position(640, 360, 1920, 1080)
    assert caplog.records[-1].meta['position'] == (960, 540), 'correct pixel position is logged'
