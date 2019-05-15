def get_counting_line(line_position, frame_width, frame_height):
    line_positions = ['top', 'bottom', 'left', 'right']
    if line_position == None:
        line_position = 'bottom'
    if line_position not in line_positions:
        raise Exception('Invalid line position specified (options: top, bottom, left, right)')

    if line_position == 'top':
        counting_line_y = round(1 / 5 * frame_height)
        return [(0, counting_line_y), (frame_width, counting_line_y)]
    elif line_position == 'bottom':
        counting_line_y = round(4 / 5 * frame_height)
        return [(0, counting_line_y), (frame_width, counting_line_y)]
    elif line_position == 'left':
        counting_line_x = round(1 / 5 * frame_width)
        return [(counting_line_x, 0), (counting_line_x, frame_height)]
    elif line_position == 'right':
        counting_line_x = round(4 / 5 * frame_width)
        return [(counting_line_x, 0), (counting_line_x, frame_height)]

def is_passed_counting_line(point, counting_line, line_position):
    if line_position == 'top':
        return point[1] < counting_line[0][1]
    elif line_position == 'bottom':
        return point[1] > counting_line[0][1]
    elif line_position == 'left':
        return point[0] < counting_line[0][0]
    elif line_position == 'right':
        return point[0] > counting_line[0][0]