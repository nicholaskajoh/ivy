import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
import numpy as np
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
import uuid
import os
import contextlib
from datetime import datetime
import argparse
from utils.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line

class VehicleCounter():

    def __init__(self, detector='yolo', tracker='kcf', droi=[], show_droi = False, mctf=3, di=10, record=False, record_path='./videos/output.avi', log_file='log.txt', cl_position='bottom'):
        self.detector = detector
        self.tracker = tracker
        self.droi =  droi
        self.show_droi = show_droi
        self.mctf = mctf
        self.detection_interval = di
        self.record = record
        self.record_destination = record_path
        self.log_file_name = log_file
        self.cl_position = cl_position
        self.is_initialized = False
        self.log_file = None
        self.output_video = None

    def initialize(self):
        if(not self.droi):
            self.frame_height, self.frame_width, _ = self.frame.shape
            self.droi = [(0, 0), (self.frame_width, 0), 
                (self.frame_width, self.frame_height), 
                    (0, self.frame_height)]
        else:
            tmp_droi = []
            points = self.droi.replace(' ', '').split('|')
            for point_str in points:
                point = tuple(map(int, point_str.split(',')))
                tmp_droi.append(point)
            self.droi = tmp_droi

        self.blobs = OrderedDict()
        self.blob_id = 1
        self.frame_counter = 0
        self.vehicle_count = 0
        self.counting_line = get_counting_line(self.cl_position, self.frame_width, self.frame_height)

    def reset(self):
        self.frame = None
        self.droi = None
        self.is_initialized = False
    
    def initialize_recording(self):
        """
        Initialises recording for a given cv frame
        """
        if self.record:
            self.output_video = cv2.VideoWriter(self.record_destination, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.frame_width, self.frame_height))
            log_file_name = 'log.txt'
            with contextlib.suppress(FileNotFoundError):
                os.remove(log_file_name)
            self.log_file = open(log_file_name, 'a')
            self.log_file.write('vehicle_id, count, datetime\n')
            self.log_file.flush()

    def initialize_blobs(self):
        """
        Given a cv frame, update blobs in relation initial blob boxes
        """
        droi_frame = get_roi_frame(self.frame, self.droi)
        initial_bboxes = get_bounding_boxes(droi_frame, self.detector)
        for box in initial_bboxes:
            _blob = create_blob(box, frame, self.tracker)
            self.blobs[self.blob_id] = _blob
            self.blob_id += 1

    def set_detector(self,detector):
        """
        Sets the detector model/algorithm to be used for vehicle counting
        options: yolo, haarc, bgsub, ssd
        """
        self.detector = detector

    def set_tracker(self, tracker):
        """
        Sets the tracker model/algorithm to be used for tracking vehicles
        options: csrt, kcf, camshift | default: kcf
        """
        self.tracker = tracker

    def set_droi(self, droi):
        """
        Set a detection region of interest (DROI) 
        i.e a set of vertices that represent the area (polygon) where
        you want detections to be made
        example: 1,2|3,4|5,6|7,8|9,10
        """
        if(droi):
            self.droi = []
            points = droi.replace(' ', '').split('|')
            for point_str in points:
                point = tuple(map(int, point_str.split(',')))
                self.droi.append(point)

    def set_show_droi(self, show_droi):
        """
        Set whether a display/overlay representing the DROI should be displayed
        """
        self.show_droi = show_droi

    def set_mctf(self, mctf):
        """
        Set the number of maximum consecutive tracking failures 
        before the tracker concludes that a tracked object has left the frame
        """
        self.mctf = mctf
    
    def set_detection_interval(self, di):
        """
        Set the number of frames before detection is carried out again 
        in order to find new vehicles and update the trackers of old ones
        """
        self.detection_interval = di

    def set_record(self, flag):
        """
        Set whether a video record and count logs should be saved to file
        """
        self.record = flag

    def set_record_destination(self,path_to_file):
        """
        Set the destination folder for recorded videos and count logs
        """
        self.record_destination = path_to_file

    def set_cl_position(self, cl_position):
        """
        Set the position for the vehicle counting line 
        """
        self.cl_position = cl_position

    def count_vehicles(self, frame):
        """
        Overlays a frame with a DROI (where necessary) and add boxes with labels
        over detected vehicles in the frame then use trackers to determine 
        vehicles that have crossed the counting line. 
        Subsequently a video record and vehicle count log files are updated if required
        """

        self.frame = frame
        if not self.is_initialized:
            self.initialize()
            self.is_initialized = True

        self.initialize_recording
        self.initialize_blobs
                
        for _id, blob in list(self.blobs.items()):
            # update trackers
            success, box = blob.tracker.update(self.frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
            else:
                blob.num_consecutive_tracking_failures += 1

            # delete untracked blobs
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[_id]

            # count vehicles
            if is_passed_counting_line(blob.centroid, self.counting_line, self.cl_position) and not blob.counted:
                blob.counted = True
                self.vehicle_count += 1

                # log count data to a file (vehicle_id, count, datetime)
                if self.record:
                    _row = '{0}, {1}, {2}\n'.format('v_' + str(_id), self.vehicle_count, datetime.now())
                    self.log_file.write(_row)
                    self.log_file.flush()

        if self.frame_counter >= self.detection_interval:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            boxes = get_bounding_boxes(droi_frame, self.detector)
            self.blobs, current_blob_id = add_new_blobs(boxes, self.blobs, self.frame, self.tracker, self.blob_id, self.counting_line, self.cl_position)
            self.blob_id = current_blob_id
            self.blobs = remove_duplicates(self.blobs)
            self.frame_counter = 0

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # draw counting line
        cv2.line(self.frame, self.counting_line[0], self.counting_line[1], (0, 255, 0), 3)

        # display vehicle count
        cv2.putText(self.frame, 'Count: ' + str(self.vehicle_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # show detection roi
        if self.show_droi:
            self.frame = draw_roi(self.frame, self.droi)

        # save frame in video output
        if self.record:
            self.output_video.write(self.frame)

        # increase frame count
        self.frame_counter += 1

        #return processed frame
        return self.frame

    def __del__(self):
        """
        Clean up release resources when this class is no longer needed
        """
        if self.log_file:
            self.log_file.close()

        if self.output_video:
            self.output_video.release()

if __name__ == '__main__':
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='relative/absolute path to video or camera input of traffic scene')
    parser.add_argument('--iscam', action='store_true', help='specify if video capture is from a camera')
    parser.add_argument('--droi', help='specify a detection region of interest (ROI) \
                        i.e a set of vertices that represent the area (polygon) \
                        where you want detections to be made (format: 1,2|3,4|5,6|7,8|9,10 \
                        default: 0,0|frame_width,0|frame_width,frame_height|0,frame_height \
                        [i.e the whole video frame])')
    parser.add_argument('--showdroi', action='store_true', help='display/overlay the detection roi on the video')
    parser.add_argument('--mctf', type=int, help='maximum consecutive tracking failures \
                        i.e number of tracking failures before the tracker concludes \
                        the tracked object has left the frame')
    parser.add_argument('--di', type=int, help='detection interval i.e number of frames \
                        before detection is carried out again (in order to find new vehicles \
                        and update the trackers of old ones)')
    parser.add_argument('--detector', help='select a model/algorithm to use for vehicle detection \
                        (options: yolo, haarc, bgsub, ssd | default: yolo)')
    parser.add_argument('--tracker', help='select a model/algorithm to use for vehicle tracking \
                        (options: csrt, kcf, camshift | default: kcf)')
    parser.add_argument('--record', action='store_true', help='record video and vehicle count logs')
    parser.add_argument('--clposition', help='position of counting line (options: top, bottom, \
                        left, right | default: bottom)')
    args = parser.parse_args()

    # capture traffic scene video
    video = int(args.video) if args.iscam else args.video
    cap = cv2.VideoCapture(video)
    vehicle_counter = VehicleCounter()

    if args.di:
        vehicle_counter.set_detection_interval(args.di)

    if args.mctf:
        vehicle_counter.set_mctf(args.mctf)

    if args.detector:
        vehicle_counter.set_detector(args.detector)

    if args.tracker:
        vehicle_counter.set_tracker(args.tracker)

    # init video object and log file to record counting
    if args.record:
        vehicle_counter.set_record(True)

    if args.clposition:
        vehicle_counter.set_cl_position(args.clposition)

    # create detection ROI
    if args.droi:
        vehicle_counter.set_droi(args.droi)
    
    if args.showdroi:
        vehicle_counter.set_show_droi(True)

    # loop through cv frame and count vvehicles
    while True:
        k = cv2.waitKey(1)
        if args.iscam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            _, cv_frame = cap.read()            
            
            # visualize vehicle counting
            vc_frame = vehicle_counter.count_vehicles(cv_frame)
            resized_frame = cv2.resize(vc_frame, (858, 480))
            cv2.imshow('tracking', resized_frame)

            # save frame if 's' key is pressed
            if k & 0xFF == ord('s'):
                cv2.imwrite(os.path.join('screenshots', 'ss_' + uuid.uuid4().hex + '.png'), vc_frame)
                print('Screenshot taken.')
        else:
            print('End of video.')
            # end video loop if on the last frame
            break

        # end video loop if 'q' key is pressed
        if k & 0xFF == ord('q'):
            print('Video exited.')
            break

    # end capture, close window, close log file and video objects if any
    cap.release()
    cv2.destroyAllWindows()
