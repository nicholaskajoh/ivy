import cv2

face_cascade = cv2.CascadeClassifier('../HaarCascades/frontalface.xml')
smile_cascade = cv2.CascadeClassifier('../HaarCascades/smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    smile = smile_cascade.detectMultiScale(gray, 1.5, 15)
    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, 'Smile', (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()