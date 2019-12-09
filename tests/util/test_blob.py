import cv2
from util.blob import Blob


def test_blob_creation():
    _bounding_box = [1, 1, 4, 4]
    _type = 'car'
    _confidence = 0.99
    _tracker = cv2.TrackerKCF_create()
    blob = Blob(_bounding_box, _type, _confidence, _tracker)
    assert isinstance(blob, Blob), 'blob is an instance of class Blob'
    assert blob.bounding_box == _bounding_box
    assert blob.type == _type
    assert blob.type_confidence == _confidence
    assert isinstance(blob.tracker, cv2.Tracker), 'blob tracker is an instance of OpenCV Tracker class'

def test_blob_update():
    _bounding_box = [1, 1, 4, 4]
    _type = 'car'
    _confidence = 0.99
    _tracker = cv2.TrackerKCF_create()
    blob = Blob(_bounding_box, _type, _confidence, _tracker)

    _new_bounding_box = [2, 2, 5, 5]
    _new_type = 'bus'
    _new_confidence = 0.35
    _new_tracker = cv2.TrackerCSRT_create()
    blob.update(_new_bounding_box, _new_type, _new_confidence, _new_tracker)

    assert blob.bounding_box == _new_bounding_box
    assert blob.type == _new_type
    assert blob.type_confidence == _new_confidence
    assert blob.tracker == _new_tracker
