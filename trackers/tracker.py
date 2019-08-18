import sys
sys.path.append('..')

from trackers.opencv.opencv_trackers import csrt_create, kcf_create
from trackers.camshift.camshift_tracker import camshift_create
from blobs.utils import get_centroid, get_area, get_iou
from counter import is_passed_counting_line
from utils.vehicle import generate_vehicle_id


def create_blob(bounding_box, vehicle_type, type_confidence, frame, model):
    if model == 'csrt':
        return csrt_create(bounding_box, vehicle_type, type_confidence, frame)
    if model == 'kcf':
        return kcf_create(bounding_box, vehicle_type, type_confidence, frame)
    if model == 'camshift':
        return camshift_create(bounding_box, vehicle_type, type_confidence, frame)
    else:
        raise Exception('Invalid tracker model/algorithm specified (options: csrt, kcf, camshift)')

def remove_stray_blobs(blobs, matched_blob_ids, mcdf):
    # remove blobs that hang after a tracked object has left the frame
    for _id, blob in list(blobs.items()):
        if _id not in matched_blob_ids:
            blob.num_consecutive_detection_failures += 1
        if blob.num_consecutive_detection_failures > mcdf:
            del blobs[_id]
    return blobs

def add_new_blobs(boxes, classes, confidences, blobs, frame, tracker, counting_line, line_position, mcdf):
    # add new blobs to existing blobs
    matched_blob_ids = []
    for _idx in range(len(boxes)):
        _type = classes[_idx] if classes != None else None
        _confidence = confidences[_idx] if confidences != None else None

        box_centroid = get_centroid(boxes[_idx])
        box_area = get_area(boxes[_idx])
        match_found = False
        for _id, blob in blobs.items():
            if blob.counted == False and get_iou(boxes[_idx], blob.bounding_box) > 0.5:
                match_found = True
                if _id not in matched_blob_ids:
                    blob.num_consecutive_detection_failures = 0
                    matched_blob_ids.append(_id)
                temp_blob = create_blob(boxes[_idx], _type, _confidence, frame, tracker) # TODO: update blob w/o creating temp blob
                blob.update(temp_blob.bounding_box, _type, _confidence, temp_blob.tracker)
                break

        if not match_found and not is_passed_counting_line(box_centroid, counting_line, line_position):
            _blob = create_blob(boxes[_idx], _type, _confidence, frame, tracker)
            blob_id = generate_vehicle_id()
            blobs[blob_id] = _blob

    blobs = remove_stray_blobs(blobs, matched_blob_ids, mcdf)
    return blobs

def remove_duplicates(blobs):
    for id_a, blob_a in list(blobs.items()):
        for id_b, blob_b in list(blobs.items()):
            if blob_a == blob_b:
                break

            if get_iou(blob_a.bounding_box, blob_b.bounding_box) > 0.5 and id_a in blobs:
                del blobs[id_a]
    return blobs