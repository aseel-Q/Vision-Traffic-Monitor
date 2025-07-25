import cv2

#upload video
cap = cv2.VideoCapture("traffic.mp4")


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.resize(frame, (800, 600))

    
    mask = object_detector.apply(frame)

    
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # تجاهل الأجسام الصغيرة
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Car", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Car Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()