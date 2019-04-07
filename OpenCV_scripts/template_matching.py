import cv2
import numpy as np

img_rgb = cv2.imread('../images/article-screenshot.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('../images/ea.jpg', 0)
w, h = template.shape[::-1]

result = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where( result >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('matches', img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()