from util.vehicle_info import generate_vehicle_id


def test_generate_vehicle_id():
    v_id = generate_vehicle_id()
    assert type(v_id) is str, 'vehicle id is a string'
    assert v_id.startswith('veh_'), "vehicle id starts with 'veh_'"
    assert len(v_id) == 4 + 32, 'vehicle id is 36 characters in length'