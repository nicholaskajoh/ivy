import uuid


def generate_object_id():
    return 'obj_' + uuid.uuid4().hex
