import sys
sys.path.append('..')

from trackers.opencv.opencv_trackers import csrt_create, kcf_create
from trackers.camshift.camshift_tracker import camshift_create
from blobs.utils import get_centroid, get_area, box_contains_point
from counter import is_passed_counting_line


def create_blob(bounding_box, frame, model):
    if model == 'csrt':
        return csrt_create(bounding_box, frame)
    if model == 'kcf':
        return kcf_create(bounding_box, frame)
    if model == 'camshift':
        return camshift_create(bounding_box, frame)
    else:
        raise Exception('Invalid tracker model/algorithm specified (options: csrt, kcf, camshift)')

def add_new_blobs(boxes, blobs, frame, tracker, current_blob_id, counting_line, line_position):
    # add new blobs to existing blobs
    for box in boxes:
        box_centroid = get_centroid(box)
        box_area = get_area(box)
        match_found = False
        for _id, blob in blobs.items():
            if blob.counted == False and \
                    ((blob.area >= box_area and box_contains_point(blob.bounding_box, box_centroid)) \
                    or (box_area >= blob.area and box_contains_point(box, blob.centroid))):
                match_found = True
                temp_blob = create_blob(box, frame, tracker) # TODO: update blob w/o creating temp blob
                blob.update(temp_blob.bounding_box, temp_blob.tracker)
                break

        if not match_found and not is_passed_counting_line(box_centroid, counting_line, line_position):
            _blob = create_blob(box, frame, tracker)
            blobs[current_blob_id] = _blob
            current_blob_id += 1
    return blobs, current_blob_id

def remove_duplicates(blobs):
    for id_a, blob_a in list(blobs.items()):
        for id_b, blob_b in list(blobs.items()):
            if blob_a == blob_b:
                break

            if blob_a.area >= blob_b.area and box_contains_point(blob_a.bounding_box, blob_b.centroid) and id_b in blobs:
                del blobs[id_b]
            elif blob_b.area >= blob_a.area and box_contains_point(blob_b.bounding_box, blob_a.centroid) and id_a in blobs:
                del blobs[id_a]
    return blobs