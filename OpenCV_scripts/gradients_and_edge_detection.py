import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    laplacian_gradient = cv2.Laplacian(frame, cv2.CV_64F)
    edges = cv2.Canny(frame, 100, 100)

    cv2.imshow('frame', frame)
    cv2.imshow('laplacian', laplacian_gradient)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()