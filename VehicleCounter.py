import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
from datetime import datetime
from utils.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line


class VehicleCounter():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, cl_position):
        self.frame = initial_frame # current frame of video
        self.detector = detector
        self.tracker = tracker
        self.droi =  droi # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.di = di # detection interval
        self.cl_position = cl_position # counting line position

        self.blobs = OrderedDict()
        self.blob_id = 1
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.vehicle_count = 0 # number of vehicles counted
        self.counting_line = None if cl_position == None else get_counting_line(self.cl_position, self.f_width, self.f_height)

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        initial_bboxes = get_bounding_boxes(droi_frame, self.detector)
        for box in initial_bboxes:
            _blob = create_blob(box, self.frame, self.tracker)
            self.blobs[self.blob_id] = _blob
            self.blob_id += 1

    def get_count(self):
        return self.vehicle_count

    def get_blobs(self):
        return self.blobs

    def count(self, frame):
        log = []
        self.frame = frame

        for _id, blob in list(self.blobs.items()):
            # update trackers
            success, box = blob.tracker.update(self.frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
            else:
                blob.num_consecutive_tracking_failures += 1

            # count vehicles that have left the frame if no counting line exists
            # or those that have passed the counting line if one exists
            if (self.counting_line == None and \
                    (blob.num_consecutive_tracking_failures == self.mctf or blob.num_consecutive_detection_failures == self.mcdf) and \
                    not blob.counted) \
                        or \
                    (self.counting_line != None and \
                    is_passed_counting_line(blob.centroid, self.counting_line, self.cl_position) and \
                    not blob.counted):
                blob.counted = True
                self.vehicle_count += 1
                log.append({'blob_id': _id, 'count': self.vehicle_count, 'datetime': datetime.now()})

            if blob.num_consecutive_tracking_failures >= self.mctf:    
                # delete untracked blobs
                del self.blobs[_id]

        if self.frame_count >= self.di:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            boxes = get_bounding_boxes(droi_frame, self.detector)
            self.blobs, current_blob_id = add_new_blobs(boxes, self.blobs, self.frame, self.tracker, self.blob_id, self.counting_line, self.cl_position, self.mcdf)
            self.blob_id = current_blob_id
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

        return log

    def visualize(self):
        frame = self.frame

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # draw counting line
        if self.counting_line != None:
            cv2.line(frame, self.counting_line[0], self.counting_line[1], (0, 255, 0), 3)
        # display vehicle count
        cv2.putText(frame, 'Count: ' + str(self.vehicle_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)

        return frame