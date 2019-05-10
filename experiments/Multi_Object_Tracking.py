import sys
sys.path.append('..')

import cv2
from blobs.blob2 import Blob
from blobs.utils import get_centroid
import numpy as np
from collections import OrderedDict

cap = cv2.VideoCapture('../videos/PeopleWalking.avi')

blobs = OrderedDict()
blob_id = 1
frame_counter = 0
DETECTION_FRAME_RATE = 48
MAX_CONSECUTIVE_TRACKING_FAILURES = 15

def get_bounding_boxes(_frame):
    fullbody_cascade = cv2.CascadeClassifier('../HaarCascades/fullbody.xml')
    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    _bounding_boxes = fullbody_cascade.detectMultiScale(gray)
    return _bounding_boxes

# initialize trackers and create new blobs
_, frame = cap.read()
initial_bboxes = get_bounding_boxes(frame)
for box in initial_bboxes:
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(box))
    _blob = Blob(box, tracker)
    blobs[blob_id] = _blob

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        _, frame = cap.read()
        
        # update trackers
        for _id, blob in list(blobs.items()):
            success, box = blob.tracker.update(frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)

                # draw and label bounding boxes
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'person' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                blob.num_consecutive_tracking_failures += 1

            if blob.num_consecutive_tracking_failures >= MAX_CONSECUTIVE_TRACKING_FAILURES:
                del blobs[_id]

        # rerun detection, add new blobs 
        if frame_counter >= DETECTION_FRAME_RATE:
            boxes = get_bounding_boxes(frame)
            
            for box in boxes:
                box_centroid = get_centroid(box)
                match_found = False
                for _id, blob in blobs.items():
                    dist = np.linalg.norm(np.array(box_centroid) - np.array(blob.centroid))
                    if dist <= 5: # 5 pixels
                        match_found = True
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, tuple(box))
                        blob.update(box, tracker)
                        break

                if not match_found:
                    blob_id += 1
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, tuple(box))
                    _blob = Blob(box, tracker)
                    blobs[blob_id] = _blob
            
            frame_counter = 0

        cv2.imshow('tracking', frame)

        frame_counter += 1
    else:
        print('End of video.')
        # end video loop if on the last frame
        break

    # end video loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Video exited.')
        break

# end capture, close window
cap.release()
cv2.destroyAllWindows()