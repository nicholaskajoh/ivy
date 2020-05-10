from util.object_info import generate_object_id


def test_generate_object_id():
    v_id = generate_object_id()
    assert type(v_id) is str, 'object id is a string'
    assert v_id.startswith('obj_'), "object id starts with 'obj_'"
    assert len(v_id) == 4 + 32, 'object id is 36 characters in length'
