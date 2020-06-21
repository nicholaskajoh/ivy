'''
Object Counter class.
'''

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import multiprocessing
import cv2
from joblib import Parallel, delayed

from tracker import add_new_blobs, remove_duplicates, update_blob_tracker
from detectors.detector import get_bounding_boxes
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from counter import attempt_count

logger = get_logger()
NUM_CORES = multiprocessing.cpu_count()

class ObjectCounter():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, counting_lines, show_counts, hud_color):
        self.frame = initial_frame # current frame of video
        self.detector = detector
        self.tracker = tracker
        self.droi = droi # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.detection_interval = di
        self.counting_lines = counting_lines
        self.blobs = {}
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts = {counting_line['label']: {} for counting_line in counting_lines} # counts of objects by type for each counting line
        self.show_counts = show_counts
        self.hud_color = hud_color

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
        self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)

    def get_counts(self):
        return self.counts

    def get_blobs(self):
        return self.blobs

    def count(self, frame):
        self.frame = frame

        blobs_list = list(self.blobs.items())
        # update blob trackers
        blobs_list = Parallel(n_jobs=NUM_CORES, prefer='threads')(
            delayed(update_blob_tracker)(blob, blob_id, self.frame) for blob_id, blob in blobs_list
        )
        self.blobs = dict(blobs_list)

        for blob_id, blob in blobs_list:
            # count object if it has crossed a counting line
            blob, self.counts = attempt_count(blob, blob_id, self.counting_lines, self.counts)
            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]

        if self.frame_count >= self.detection_interval:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)

            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

    def visualize(self):
        frame = self.frame
        font = cv2.FONT_HERSHEY_DUPLEX
        line_type = cv2.LINE_AA

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.hud_color, 2)
            object_label = 'I: ' + _id[:8] \
                            if blob.type is None \
                            else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
            cv2.putText(frame, object_label, (x, y - 5), font, 1, self.hud_color, 2, line_type)

        # draw counting lines
        for counting_line in self.counting_lines:
            cv2.line(frame, counting_line['line'][0], counting_line['line'][1], self.hud_color, 3)
            cl_label_origin = (counting_line['line'][0][0], counting_line['line'][0][1] + 35)
            cv2.putText(frame, counting_line['label'], cl_label_origin, font, 1, self.hud_color, 2, line_type)

        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)

        # show counts
        if self.show_counts:
            offset = 1
            for line, objects in self.counts.items():
                cv2.putText(frame, line, (10, 40 * offset), font, 1, self.hud_color, 2, line_type)
                for label, count in objects.items():
                    offset += 1
                    cv2.putText(frame, "{}: {}".format(label, count), (10, 40 * offset), font, 1, self.hud_color, 2, line_type)
                offset += 2

        return frame
