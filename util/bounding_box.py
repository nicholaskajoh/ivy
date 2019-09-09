def get_box_image(frame, bounding_box):
    x, y, w, h = list(map(int, bounding_box))
    return frame[y - 10:y + h + 10, x - 10:x + w + 10] # allowance of 10 pixels on every side