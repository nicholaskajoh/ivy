import logging
from unittest.mock import patch
from util.debugger import capture_pixel_position


@patch.dict('os.environ', {'DEBUG_WINDOW_SIZE': '(1280, 720)'})
def test_capture_pixel_position(caplog):
    caplog.set_level(logging.INFO)
    capture_pixel_position(640, 360, 1920, 1080)
    assert caplog.records[-1].meta['position'] == (960, 540), 'correct pixel position is logged'
