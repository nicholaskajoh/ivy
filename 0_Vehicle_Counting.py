import cv2
from blobs.blob3 import Blob, get_centroid
import numpy as np
from collections import OrderedDict
from detectors.yolo_detector import get_bounding_boxes

cap = cv2.VideoCapture('./videos/sample_traffic_scene.mp4') # ./CarsDrivingUnderBridge.mp4

blobs = OrderedDict()
blob_id = 1
frame_counter = 0
DETECTION_FRAME_RATE = 48
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

def create_blob(_box):
    x, y, w, h = [int(v) for v in _box]
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    return Blob((x, y, w, h), roi_hist)

# initialize trackers and create new blobs
_, frame = cap.read()
initial_bboxes = get_bounding_boxes(frame)
for box in initial_bboxes:
    _blob = create_blob(box)
    blobs[blob_id] = _blob

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        _, frame = cap.read()
        output_frame = frame.copy()
        
        # update camshift tracker
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for _id, blob in list(blobs.items()):
            mask = cv2.calcBackProject([hsv], [0], blob.roi_hist, [0, 180], 1)
            ret, _ = cv2.CamShift(mask, blob.roi_box, term_criteria)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            bbox = cv2.boundingRect(pts)
            x, y, w, h = [int(v) for v in bbox]

            _roi = frame[y:y + h, x:x + w]
            _hsv_roi = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV)
            _roi_hist = cv2.calcHist([_hsv_roi], [0], None, [180], [0, 180])
            correlation = cv2.compareHist(_roi_hist, blob.roi_hist, cv2.HISTCMP_CORREL)
            print('correlation', correlation)
            if (correlation >= 0.5):
                # blob.bbox = bbox

                # draw and label bounding boxes
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output_frame, 'person' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        # rerun detection, add new blobs 
        if frame_counter >= DETECTION_FRAME_RATE:
            boxes = get_bounding_boxes(frame)

            for box in boxes:
                box_centroid = get_centroid(box)
                match_found = False
                for _id, blob in blobs.items():
                    dist = np.linalg.norm(np.array(box_centroid) - np.array(blob.centroid))
                    if dist <= 50: # 5 pixels
                        match_found = True
                        break

                if not match_found:
                    blob_id += 1
                    _blob = create_blob(box)
                    blobs[blob_id] = _blob
            
            frame_counter = 0        

        resized_frame = cv2.resize(output_frame, (858, 480))
        cv2.imshow('tracking', resized_frame)

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