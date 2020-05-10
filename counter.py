# pylint: disable=missing-module-docstring,invalid-name

import time
from util.logger import get_logger


logger = get_logger()

def _line_segments_intersect(line1, line2):
    '''
    See: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    '''
    def get_orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def is_on_segment(p, q, r):
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    p1 = line1[0]
    q1 = line1[1]
    p2 = line2[0]
    q2 = line2[1]

    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and is_on_segment(p1, p2, q1):
        return True

    if o2 == 0 and is_on_segment(p1, q2, q1):
        return True

    if o3 == 0 and is_on_segment(p2, p1, q2):
        return True

    if o4 == 0 and is_on_segment(p2, q1, q2):
        return True

    return False

def _has_crossed_counting_line(bbox, line):
    '''
    Check if at least one edge of a bounding box is intersected by a counting line.
    '''
    x, y, w, h = bbox
    bbox_line1 = [(x, y), (x + w, y)]
    bbox_line2 = [(x + w, y), (x + w, y + h)]
    bbox_line3 = [(x, y), (x, y + h)]
    bbox_line4 = [(x, y + h), (x + w, y + h)]

    if _line_segments_intersect(bbox_line1, line) or \
            _line_segments_intersect(bbox_line2, line) or \
            _line_segments_intersect(bbox_line3, line) or \
            _line_segments_intersect(bbox_line4, line):
        return True
    return False

def attempt_count(blob, blob_id, counting_lines, counts):
    '''
    Check if a blob has crossed a counting line.
    '''
    for counting_line in counting_lines:
        label = counting_line['label']
        if _has_crossed_counting_line(blob.bounding_box, counting_line['line']) and \
                label not in blob.lines_crossed:
            if blob.type in counts[label]:
                counts[label][blob.type] += 1
            else:
                counts[label][blob.type] = 1

            blob.lines_crossed.append(label)

            logger.info('Object counted.', extra={
                'meta': {
                    'label': 'OBJECT_COUNT',
                    'id': blob_id,
                    'type': blob.type,
                    'counting_line': label,
                    'position_first_detected': blob.position_first_detected,
                    'position_counted': blob.centroid,
                    'counted_at':time.time(),
                },
            })
    return blob, counts
