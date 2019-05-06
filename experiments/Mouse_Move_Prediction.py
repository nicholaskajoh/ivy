import cv2
import numpy as np

mouse_position = [0, 0]
mouse_positions = []

def set_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position[0], mouse_position[1] = (x, y)

def predict_next_position(previous_mouse_positions):
    n = len(previous_mouse_positions)
    if n == 0:
        return [0, 0]
    if n == 1:
        return previous_mouse_positions[0]
    else:
        delta_xs = []
        delta_ys = []
        j = n - 10 if n - 10 > 0 else 0 # calculate deltas from a max of 10 prev positions
        c = n - 1 - j;
        for i in range(n - 1, j, -1):
            delta_x = (previous_mouse_positions[i][0] - previous_mouse_positions[i - 1][0]) * c
            delta_xs.append(delta_x)
            delta_y = (previous_mouse_positions[i][1] - previous_mouse_positions[i - 1][1]) * c
            delta_ys.append(delta_y)
            c -= 1
        average_delta_x = sum(delta_xs) / float(len(delta_xs))
        average_delta_y = sum(delta_ys) / float(len(delta_ys))
        return (
            round(previous_mouse_positions[n - 1][0] + average_delta_x),
            round(previous_mouse_positions[n - 1][1] + average_delta_y),
        )

def draw_cross(image, position, color):
    cv2.line(image, (position[0] - 5, position[1] - 5), (position[0] + 5, position[1] + 5), color, 2)
    cv2.line(image, (position[0] + 5, position[1] - 5), (position[0] - 5, position[1] + 5), color, 2)

blank = np.zeros((500, 500, 3), np.uint8)
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('frame', set_mouse_position)

while True:
    mouse_positions.append(list(mouse_position))

    predicted_mouse_position = predict_next_position(mouse_positions)

    draw_cross(blank, mouse_position, (255, 255, 255))
    draw_cross(blank, predicted_mouse_position, (255, 0, 0))

    cv2.imshow('frame', blank)

    blank = np.zeros((500, 500, 3), np.uint8)

    # end program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exited.')
        break

cv2.destroyAllWindows()