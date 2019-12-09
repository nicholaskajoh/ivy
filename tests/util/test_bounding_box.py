from util.bounding_box import get_centroid, box_contains_point, get_area


def test_get_centroid():
    bounding_box = [1, 1, 4, 4]
    centroid = get_centroid(bounding_box)
    assert type(centroid) is tuple, 'centroid is a tuple'
    assert len(centroid) == 2, 'centroid is a 2d coordinate (x, y)'
    assert centroid[0] == 3 and centroid[1] == 3, 'the centroid (center point) of box [1, 1, 4, 4] is (3, 3)'

def test_box_contains_point():
    bounding_box = [1, 1, 4, 4]
    point1 = (2, 2)
    point2 = (0, 0)
    contains_point1 = box_contains_point(bounding_box, point1)
    contains_point2 = box_contains_point(bounding_box, point2)
    assert type(contains_point1) is bool and type(contains_point2) is bool, 'return type is boolean'
    assert contains_point1 == True, 'box [1, 1, 4, 4] contains point (2, 2)'
    assert contains_point2 == False, 'box [1, 1, 4, 4] does not contain point (0, 0)'

def test_get_area():
    bounding_box = [1, 1, 4, 4]
    area = get_area(bounding_box)
    assert area == 16, 'area of box [1, 1, 4, 4] is 16'