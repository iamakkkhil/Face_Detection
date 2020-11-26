import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


face_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_frontalface_default.xml")


def adj_detect_face(img):
    
    face_img = img.copy()
    
#     It returns coordinates in x,y,w,h
    face_rectangle = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10)
        
    return face_img


# LIVESTREAMING
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    
    frame = detect_eye(frame)
    
    cv2.imshow("FACE_DETECT", frame)
    
    k = cv2.waitKey(1)
    
    if k ==27:
        break

cap.release()
cv2.destroyAllWindows()