import cv2

img = cv2.imread('../images/me.jpg', cv2.IMREAD_COLOR)
cv2.line(img, (0, 0), (150, 150), (0, 255, 255), 5)
cv2.rectangle(img, (20, 20), (300, 300), (255, 0, 255), 10)
cv2.circle(img, (350, 350), 100, (255, 255, 0), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Nicholas Kajoh', (5, 400), font, 2, (0, 0, 255), 5)

cv2.imshow('Nicholas Kajoh', img)
cv2.waitKey(0)
cv2.destroyAllWindows()