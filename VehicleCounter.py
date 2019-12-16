import cv2
from tracker import add_new_blobs, remove_duplicates
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
import time
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from counter import has_crossed_counting_line


logger = get_logger()

class VehicleCounter():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, counting_lines):
        self.frame = initial_frame # current frame of video
        self.detector = detector
        self.tracker = tracker
        self.droi = droi # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.di = di # detection interval
        self.counting_lines = counting_lines

        self.blobs = OrderedDict()
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts_by_type_per_line = {counting_line['label']: {} for counting_line in counting_lines} # counts of vehicles by type for each counting line

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
        self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)

    def get_blobs(self):
        return self.blobs

    def count(self, frame):
        self.frame = frame

        for _id, blob in list(self.blobs.items()):
            # update trackers
            success, box = blob.tracker.update(self.frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
                logger.debug('Vehicle tracker updated.', extra={
                    'meta': {
                        'label': 'TRACKER_UPDATE',
                        'vehicle_id': _id,
                        'bounding_box': blob.bounding_box,
                        'centroid': blob.centroid,
                    },
                })
            else:
                blob.num_consecutive_tracking_failures += 1

            # count vehicle if it has crossed a counting line
            for counting_line in self.counting_lines:
                label = counting_line['label']
                if has_crossed_counting_line(blob.bounding_box, counting_line['line']) and \
                        label not in blob.lines_crossed:
                    if blob.type in self.counts_by_type_per_line[label]:
                        self.counts_by_type_per_line[label][blob.type] += 1
                    else:
                        self.counts_by_type_per_line[label][blob.type] = 1

                    blob.lines_crossed.append(label)

                    logger.info('Vehicle counted.', extra={
                        'meta': {
                            'label': 'VEHICLE_COUNT',
                            'id': _id,
                            'type': blob.type,
                            'counting_line': label,
                            'position_first_detected': blob.position_first_detected,
                            'position_counted': blob.centroid,
                            'counted_at':time.time(),
                            'counts_by_type_per_line': self.counts_by_type_per_line,
                        },
                    })

            if blob.num_consecutive_tracking_failures >= self.mctf:
                # delete untracked blobs
                del self.blobs[_id]

        if self.frame_count >= self.di:
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            vehicle_label = 'I: ' + _id[:8] \
                            if blob.type is None \
                            else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
            cv2.putText(frame, vehicle_label, (x, y - 5), font, 1, (255, 0, 0), 2, line_type)

        # draw counting lines
        for counting_line in self.counting_lines:
            cv2.line(frame, counting_line['line'][0], counting_line['line'][1], (255, 0, 0), 3)
            cl_label_origin = (counting_line['line'][0][0], counting_line['line'][0][1] + 35)
            cv2.putText(frame, counting_line['label'], cl_label_origin, font, 1, (255, 0, 0), 2, line_type)

        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)

        return frame
