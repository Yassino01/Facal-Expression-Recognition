import cv2
from PIL import Image


# To capture video from webcam.

#test = face_cascade.load('haarcascade_frontalface_default.xml')

#print(test)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_cascade)

cap = cv2.VideoCapture(0)
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    img = cv2.putText(img, "Le VTC", org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('img', img)
    # Stop if (Q) key is pressed
    k = cv2.waitKey(30)
    if k==ord("q"):
        print("quit")
        break

# Release the VideoCapture object
cap.release()