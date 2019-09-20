import cv2
from util.bounding_box import get_centroid, get_area, get_iou, get_box_image
from util.blob import Blob
from counter import is_passed_counting_line
from util.vehicle_info import generate_vehicle_id
from util.logger import log_info
from util.image import get_base64_image


def csrt_create(bounding_box, frame):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker

def kcf_create(bounding_box, frame):
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker

def get_tracker(algorithm, bounding_box, frame):
    if algorithm == 'csrt':
        return csrt_create(bounding_box, frame)
    if algorithm == 'kcf':
        return kcf_create(bounding_box, frame)
    else:
        raise Exception('Invalid tracking algorithm specified (options: csrt, kcf)')

def remove_stray_blobs(blobs, matched_blob_ids, mcdf):
    # remove blobs that hang after a tracked object has left the frame
    for _id, blob in list(blobs.items()):
        if _id not in matched_blob_ids:
            blob.num_consecutive_detection_failures += 1
        if blob.num_consecutive_detection_failures > mcdf:
            del blobs[_id]
    return blobs

def add_new_blobs(boxes, classes, confidences, blobs, frame, tracker, counting_line, line_position, mcdf):
    # add new blobs or update existing ones
    matched_blob_ids = []
    for i in range(len(boxes)):
        _type = classes[i] if classes != None else None
        _confidence = confidences[i] if confidences != None else None
        _tracker = get_tracker(tracker, boxes[i], frame)

        box_centroid = get_centroid(boxes[i])
        box_area = get_area(boxes[i])
        match_found = False
        for _id, blob in blobs.items():
            if blob.counted == False and get_iou(boxes[i], blob.bounding_box) > 0.5:
                match_found = True
                if _id not in matched_blob_ids:
                    blob.num_consecutive_detection_failures = 0
                    matched_blob_ids.append(_id)
                blob.update(boxes[i], _type, _confidence, _tracker)

                log_info('Blob updated.', {
                    'cat': 'BLOB_UPSERT',
                    'vehicle_id': _id,
                    'bounding_box': blob.bounding_box,
                    'type': blob.type,
                    'type_confidence': blob.type_confidence,
                    'image': get_base64_image(get_box_image(frame, blob.bounding_box))
                })
                break

        if not match_found and not is_passed_counting_line(box_centroid, counting_line, line_position):
            _blob = Blob(boxes[i], _type, _confidence, _tracker)
            blob_id = generate_vehicle_id()
            blobs[blob_id] = _blob

            log_info('Blob created.', {
                'cat': 'BLOB_UPSERT',
                'vehicle_id': blob_id,
                'bounding_box': _blob.bounding_box,
                'type': _blob.type,
                'type_confidence': _blob.type_confidence,
                'image': get_base64_image(get_box_image(frame, _blob.bounding_box))
            })

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