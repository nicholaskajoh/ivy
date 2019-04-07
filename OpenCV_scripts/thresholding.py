import cv2

img = cv2.imread('../images/bookpage.jpg', cv2.IMREAD_COLOR)

retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval_gs, threshold_gs = cv2.threshold(greyscaled_img, 12, 255, cv2.THRESH_BINARY)

g_threshold = cv2.adaptiveThreshold(greyscaled_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# cv2.imshow('Book page', img)
cv2.imshow('Book page threshold', threshold)
# cv2.imshow('Book page greyscaled', greyscaled_img)
cv2.imshow('Book page greyscaled threshold', threshold_gs)
cv2.imshow('Book page greyscaled gaussian threshold', g_threshold)

# press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()