import numpy as np
import cv2

img = cv2.imread('../images/AlphaCoderLogo-white.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 15, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

cv2.imshow('corners', img)

cv2.waitKey(0)
cv2.destroyAllWindows()