# Vision-Traffic-Monitor
VisionTraffic Monitor

Project Overview:

In this project, vehicle movement was tracked using the OpenCV library by analyzing a video of a busy street. The program relies on background detection technology to isolate moving objects, thereby identifying vehicles and tracking their movement in real-time within the video.

This type of project is used in various fields, such as surveillance systems and traffic analysis, and is a fundamental step toward developing intelligent systems capable of automatically monitoring roads.

Benefits of the Project:

The project helps track vehicle movement automatically, which can be useful for road monitoring, recording vehicle counts, or even analyzing traffic congestion. The idea is simple, but it can be developed and used in larger systems such as smart cameras or traffic systems, especially in areas that require constant monitoring of vehicle movement.

Project Creation Steps:

1. First, I downloaded Anaconda.
2. Then, I downloaded the OpenCV library.
3. Then, I ran VS Code and wrote Python code to train the model. I placed it in a special folder with the video clip.
4. I opened the project folder and ran it. The expected results of the project, such as tracking and identifying cars within a frame, appeared.
5. Model training code:

import cv2

# upload video 
cap = cv2.VideoCapture("traffic.mp4")

# Create a background for subtraction technique
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

  # Resize to speed up processing
    frame = cv2.resize(frame, (800, 600))

  # Extract objects from the background
    mask = object_detector.apply(frame)

  # Image cleaning
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw boxes around cars
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:  # Ignore small objects
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Car", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Car Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
