import cv2
import numpy as np


def find_head (image):
    face_cascade = cv2.CascadeClassifier('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
