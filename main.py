'''
VCS entry point.
'''

import ast
import os

import cv2
from dotenv import load_dotenv

from VehicleCounter import VehicleCounter
from util.logger import log_info, log_error
from util.image import take_screenshot


def run():
    '''
    Loads environment variables, initializes counter class and runs counting loop.
    '''
    # capture traffic scene video
    is_cam = ast.literal_eval(os.getenv('IS_CAM'))
    video = int(os.getenv('VIDEO')) if is_cam else os.getenv('VIDEO')
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        log_error('Error capturing video. Invalid source.', {'cat': 'VIDEO_CAPTURE', 'source': video})
    ret, frame = cap.read()
    f_height, f_width, _ = frame.shape

    detection_interval = int(os.getenv('DI'))
    mcdf = int(os.getenv('MCDF'))
    mctf = int(os.getenv('MCTF'))
    detector = os.getenv('DETECTOR')
    tracker = os.getenv('TRACKER')
    # create detection region of interest polygon
    use_droi = ast.literal_eval(os.getenv('USE_DROI'))
    droi = ast.literal_eval(os.getenv('DROI')) \
            if use_droi \
            else [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    show_droi = ast.literal_eval(os.getenv('SHOW_DROI'))
    counting_line_position = os.getenv('COUNTING_LINE_POSITION')

    vehicle_counter = VehicleCounter(frame, detector, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_line_position)

    record = ast.literal_eval(os.getenv('RECORD'))
    headless = ast.literal_eval(os.getenv('HEADLESS'))

    if record:
        # initialize video object to record counting
        output_video = cv2.VideoWriter(os.getenv('OUTPUT_VIDEO_PATH'), \
                                        cv2.VideoWriter_fourcc(*'MJPG'), \
                                        30, \
                                        (f_width, f_height))

    log_info('Processing started...',
             {'cat': 'COUNT_PROCESS',
              'counter_config': {'di': detection_interval, 'mcdf': mcdf, 'mctf': mctf, 'detector': detector,
                                 'tracker': tracker, 'use_droi': use_droi, 'droi': droi, 'show_droi': show_droi,
                                 'clp': counting_line_position}})
    # main loop
    while is_cam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        if ret:
            vehicle_counter.count(frame)
            output_frame = vehicle_counter.visualize()

            if record:
                output_video.write(output_frame)

            if not headless:
                debug_window_size = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
                resized_frame = cv2.resize(output_frame, debug_window_size)
                cv2.imshow('Debug', resized_frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'): # save frame if 's' key is pressed
            take_screenshot(output_frame)
        if k == ord('q'): # end video loop if 'q' key is pressed
            log_info('Processing stopped.', {'cat': 'COUNT_PROCESS'})
            break

        ret, frame = cap.read()

    # end capture, close window, close log file and video object if any
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    log_info('Processing ended.', {'cat': 'COUNT_PROCESS'})

if __name__ == '__main__':
    load_dotenv()
    run()
