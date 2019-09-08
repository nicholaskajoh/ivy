import uuid


def generate_vehicle_id():
    return 'veh_' + uuid.uuid4().hex