import cv2

cap = cv2.VideoCapture('../videos/sample_traffic_scene.mp4')

while True:
    ret, frame = cap.read()
    cv2.imshow('video capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()