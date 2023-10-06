import cv2
import imutils
import numpy as np
import os
import math
from statistics import mean
from roifind import roi_find
from json_data_out import get_points
from find_head import find_head
import main_working_2

image = cv2.imread("mecd.jpg")
roi_find ('mecd.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),8)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#Contours
contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    if cv2.contourArea(contour) < 3000:  # for removing noises
        continue
        
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)

cv2.imshow('Image', mask)
cv2.waitKey(0)
