import cv2

img = cv2.imread('../images/me.jpg', cv2.IMREAD_COLOR)

pixel = img[50, 50]
print(pixel)

region = img[100:150, 100:150]
img[200:300, 200:300] = [255, 255, 255]

cv2.imshow('Nicholas Kajoh', img)
# press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()