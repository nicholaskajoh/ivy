import cv2

cap = cv2.VideoCapture('../videos/sample_traffic_scene.mp4')
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    bg_subtractor_mask = bg_subtractor.apply(frame)

    cv2.imshow('video', frame)
    cv2.imshow('mask', bg_subtractor_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()