'''
VCS entry point.
'''

def run():
    '''
    Initialize counter class and run counting loop.
    '''

    import ast
    import os
    import sys
    import time

    import cv2

    from util.image import take_screenshot
    from util.logger import get_logger
    from util.debugger import mouse_callback
    from VehicleCounter import VehicleCounter

    logger = get_logger()

    # capture traffic scene video
    is_cam = ast.literal_eval(os.getenv('IS_CAM'))
    video = int(os.getenv('VIDEO')) if is_cam else os.getenv('VIDEO')
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception('Invalid video source {0}'.format(video))
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
    counting_lines = ast.literal_eval(os.getenv('COUNTING_LINES'))

    vehicle_counter = VehicleCounter(frame, detector, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_lines)

    record = ast.literal_eval(os.getenv('RECORD'))
    headless = ast.literal_eval(os.getenv('HEADLESS'))

    if record:
        # initialize video object to record counting
        output_video = cv2.VideoWriter(os.getenv('OUTPUT_VIDEO_PATH'), \
                                        cv2.VideoWriter_fourcc(*'MJPG'), \
                                        30, \
                                        (f_width, f_height))

    logger.info('Processing started.', extra={
        'meta': {
            'label': 'START_PROCESS',
            'counter_config': {
                'di': detection_interval,
                'mcdf': mcdf,
                'mctf': mctf,
                'detector': detector,
                'tracker': tracker,
                'use_droi': use_droi,
                'droi': droi,
                'show_droi': show_droi,
                'counting_lines': counting_lines
            },
        },
    })

    if not headless:
        # capture mouse events in the debug window
        cv2.namedWindow('Debug')
        cv2.setMouseCallback('Debug', mouse_callback, {'frame_width': f_width, 'frame_height': f_height})

    is_paused = False
    output_frame = None

    # main loop
    while is_cam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'): # pause/play loop if 'p' key is pressed
            is_paused = False if is_paused else True
            logger.info('Loop paused/played.', extra={'meta': {'label': 'PAUSE_PLAY_LOOP', 'is_paused': is_paused}})
        if k == ord('s') and output_frame is not None: # save frame if 's' key is pressed
            take_screenshot(output_frame)
        if k == ord('q'): # end video loop if 'q' key is pressed
            logger.info('Loop stopped.', extra={'meta': {'label': 'STOP_LOOP'}})
            break

        if is_paused:
            time.sleep(0.5)
            continue

        _timer = cv2.getTickCount() # set timer to calculate processing frame rate

        if ret:
            vehicle_counter.count(frame)
            output_frame = vehicle_counter.visualize()

            if record:
                output_video.write(output_frame)

            if not headless:
                debug_window_size = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
                resized_frame = cv2.resize(output_frame, debug_window_size)
                cv2.imshow('Debug', resized_frame)

        processing_frame_rate = round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 2)
        frames_processed = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frames_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug('Frame processed.', extra={
            'meta': {
                'label': 'FRAME_PROCESS',
                'frames_processed': frames_processed,
                'frame_rate': processing_frame_rate,
                'frames_left': frames_count - frames_processed,
                'percentage_processed': round((frames_processed / frames_count) * 100, 2),
            },
        })

        ret, frame = cap.read()

    # end capture, close window, close log file and video object if any
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    logger.info('Processing ended.', extra={'meta': {'label': 'END_PROCESS'}})


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    from util.logger import init_logger
    init_logger()

    run()
