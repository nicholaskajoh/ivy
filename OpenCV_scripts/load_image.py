import cv2

img = cv2.imread('../images/me.jpg', cv2.IMREAD_GRAYSCALE) # alternatively IMREAD_COLOR or IMREAD_UNCHANGED
cv2.imshow('Nicholas Kajoh', img)

# press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image
# cv2.imwrite('me-shouting-in-grayscale.jpg', img)