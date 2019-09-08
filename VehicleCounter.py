import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
import time
from util.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line
from util.vehicle_info import generate_vehicle_id
from util.logger import log_info


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
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.vehicle_count = 0 # number of vehicles counted
        self.types_counts = OrderedDict() # counts by vehicle type
        self.counting_line = None if cl_position == None else get_counting_line(self.cl_position, self.f_width, self.f_height)

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
        for _idx in range(len(_bounding_boxes)):
            _type = _classes[_idx] if _classes != None else None
            _confidence = _confidences[_idx] if _confidences != None else None
            _blob = create_blob(_bounding_boxes[_idx], _type, _confidence, self.frame, self.tracker)
            blob_id = generate_vehicle_id()
            self.blobs[blob_id] = _blob

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
                log_info('Vehicle tracker updated.', {
                    'event': 'TRACKER_UPDATE',
                    'vehicle_id': _id,
                    'bounding_box': blob.bounding_box,
                    'centroid': blob.centroid,
                })
            else:
                blob.num_consecutive_tracking_failures += 1

            # count vehicles that have left the frame if no counting line exists
            # or those that have passed the counting line if one exists
            if (self.counting_line == None and \
                    (blob.num_consecutive_tracking_failures == self.mctf or blob.num_consecutive_detection_failures == self.mcdf) and \
                    not blob.counted) \
                        or \
                    (self.counting_line != None and \
                    # don't count a blob if it was first detected at a position past the counting line
                    # this enforces counting in only one direction
                    not is_passed_counting_line(blob.first_detected_at, self.counting_line, self.cl_position) and \
                    is_passed_counting_line(blob.centroid, self.counting_line, self.cl_position) and \
                    not blob.counted):
                blob.counted = True
                self.vehicle_count += 1
                # count by vehicle type
                if blob.type != None:
                    if blob.type in self.types_counts:
                        self.types_counts[blob.type] += 1
                    else:
                        self.types_counts[blob.type] = 1
                log_info('Vehicle counted.', {
                    'event': 'VEHICLE_COUNT',
                    'id': _id,
                    'type': blob.type,
                    'count': self.vehicle_count,
                    'counted_at':time.time(),
                })

            if blob.num_consecutive_tracking_failures >= self.mctf:    
                # delete untracked blobs
                del self.blobs[_id]

        if self.frame_count >= self.di:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.counting_line, self.cl_position, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

        return log

    def visualize(self):
        frame = self.frame

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            vehicle_label = 'ID: ' + _id[:8] \
                            if blob.type == None \
                            else 'ID: {0}, Type: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
            cv2.putText(frame, vehicle_label, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
        # draw counting line
        if self.counting_line != None:
            cv2.line(frame, self.counting_line[0], self.counting_line[1], (255, 0, 0), 3)
        # display vehicle count
        types_counts_str = ', '.join([': '.join(map(str, i)) for i in self.types_counts.items()])
        types_counts_str = ' (' + types_counts_str + ')' if types_counts_str != '' else types_counts_str
        cv2.putText(frame, 'Count: ' + str(self.vehicle_count) + types_counts_str, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)

        return frame