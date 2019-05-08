import sys
sys.path.append('..')

import cv2
from blobs.blob import Blob

cap = cv2.VideoCapture('../videos/PeopleWalking.avi')
# get first 2 frames
_, image1 = cap.read()
_, image2 = cap.read()

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # display original input image
        cv2.imshow('original', image1)

        # convert images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # blur images to remove noise
        image1_blur = cv2.GaussianBlur(image1_gray, (5, 5), 0)
        image2_blur = cv2.GaussianBlur(image2_gray, (5, 5), 0)

        # get the absolute difference between the 2 images
        abs_diff_image = cv2.absdiff(image1_blur, image2_blur)

        # add threshold to make objects clearer
        _, threshold_image = cv2.threshold(abs_diff_image, 30, 255, cv2.THRESH_BINARY)

        # erode and dilate threshold image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        threshold_image = cv2.dilate(threshold_image, kernel)
        threshold_image = cv2.dilate(threshold_image, kernel)
        threshold_image = cv2.erode(threshold_image, kernel)

        # cv2.imshow('threshold image', threshold_image)

        # find and draw contours on image
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(threshold_image, contours, -1, (255, 255, 255), -1)

        cv2.imshow('contours', threshold_image)

        # get convex hulls from contours
        convex_hulls = []
        for i in range(len(contours)):
            convex_hulls.append(cv2.convexHull(contours[i], False))

        blobs = []
        # filter out blobs that are not sizeable enough to be an object (e.g human, vehicle etc)
        for convex_hull in convex_hulls:
            possible_blob = Blob(convex_hull)
            if possible_blob.bounding_rect['area'] > 100 and \
                    possible_blob.aspect_ratio >= 0.2 and \
                    possible_blob.aspect_ratio <= 1.2 and \
                    possible_blob.bounding_rect['w'] > 15 and \
                    possible_blob.bounding_rect['h'] > 20 and \
                    possible_blob.diagonal_size > 30.0:
                blobs.append(possible_blob)

        # get convex hulls from blobs
        blob_convex_hulls = []
        for blob in blobs:
            blob_convex_hulls.append(blob.convex_hull)

        cv2.drawContours(threshold_image, blob_convex_hulls, -1, (255, 255, 255), -1)

        cv2.imshow('convex hulls', threshold_image)

        for blob in blobs:
            rect_top_left = (blob.bounding_rect['x'], blob.bounding_rect['y'])
            rect_bottom_right = (
                blob.bounding_rect['x'] + blob.bounding_rect['w'],
                blob.bounding_rect['y'] + blob.bounding_rect['h'],
            )
            cv2.rectangle(image1, rect_top_left, rect_bottom_right, (0, 0, 255), 2)
            circle_center = (blob.center_position['x'], blob.center_position['y'])
            circle_radius = 4
            cv2.circle(image1, circle_center, circle_radius, (0, 255, 0), -1)

        cv2.imshow('bounding boxes', image1)


        # get next consecutive frames
        image1 = image2
        _, image2 = cap.read()
    else:
        print('End of video.')
        # end video loop if on the last frame
        break

    # end video loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Video exited.')
        break

# end capture, close window
cap.release()
cv2.destroyAllWindows()