import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([150, 150, 50])
    upper_bound = np.array([180, 255, 150])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    out_frame = cv2.bitwise_and(frame, frame, mask=mask)
    median_blur = cv2.medianBlur(out_frame, 15)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('output frame', out_frame)
    cv2.imshow('median blur', median_blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()