import cv2
import imutils
import numpy as np
import os
import math
from roifind import roi_find
from json_data_out import get_points
#from find_head import find_head
#from statistics import mean

# read the image
image = cv2.imread("mecd.jpg")

# ROI - detect a peson to be measured
roi_find ('mecd.jpg')
x, y, w, h = roi_find('mecd.jpg')
#image = image[y:y+h, x:x+w] 


#user_height = int(input('Input your height:')) # y+h
user_height = 176

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


#cv2.imshow('Mask', mask)

# ROI
image = image[y:y+h, x:x+w]
height = image.shape[0]
width = image.shape[1]

#Run openpose
image_dir = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement'
json_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement'
image_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement'

os.chdir('/home/serhii/openpose')
print(os.getcwd())
#os.system('./build/examples/openpose/openpose.bin --net_resolution -1x128 --image_dir /home/serhii/Documents/PyProjects/Body-measurements/scripting-2 --write_json /home/serhii/Documents/PyProjects/Body-measurements/scripting-2/ --write_images /home/serhii/Documents/PyProjects/Body-measurements/scripting-2/')
os.system('./build/examples/openpose/openpose.bin --net_resolution -1x128 --image_dir ' + image_dir + ' --write_json ' + json_out +' --write_images ' + image_out)
os.chdir('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/')
print(os.getcwd())

# Getting points from openpose
points = get_points('mecd_keypoints.json')
i = 0
points_group = []
while i <= len(points) - 3:
    a = [points[i],points[i+1],points[i+2]]
    points_group.append(a)
    i += 3

#print(points_group)

# Finding xs and ys
point_x = []
point_y = []
for point in points_group:
    for x in point:
        if x == point[0]:
           point_x.append(x)
        elif x == point[1]:
            point_y.append(x)
        else:
            pass

#print (point_x)
#print (point_y)


#Find the lenght of the arm, using the openpose points example
def hands_measurements (hand_points_x, hand_points_y, body_size_cm, body_size_coordiantes, **img): # ..., user_height, y+h
    
    # sqrt ((x1-x0)^2 + (y1-y0)^2)
    length_v_x_1 = (int(hand_points_x[0])- int(hand_points_x[1]))**2
    length_v_y_1 = (int(hand_points_y[0])- int(hand_points_y[1]))**2
    length_v_1 = abs(length_v_x_1 + length_v_y_1)
    length_v_1 = math.sqrt(length_v_1)
    #print (length_v_1)

    length_v_x_2 = (int(hand_points_x[2])- int(hand_points_x[1]))**2
    length_v_y_2 = (int(hand_points_y[2])- int(hand_points_y[1]))**2
    length_v_2 = abs(length_v_x_2 + length_v_y_2)
    length_v_2 = math.sqrt(length_v_2)
    #print (length_v_2)

    lenght_of_hand = length_v_1 + length_v_2
    hand_in_cm = (lenght_of_hand * body_size_cm)/(body_size_coordiantes)
    #print (hand_in_cm)

    img = cv2.imread('mecd.jpg')

    img = cv2.circle(img, (int(hand_points_x[0]), int(hand_points_y[0])), 3, (200, 100, 50), 3)
    img = cv2.circle(img, (int(hand_points_x[1]), int(hand_points_y[1])), 3, (200, 100, 50), 3)
    img = cv2.circle(img, (int(hand_points_x[2]), int(hand_points_y[2])), 3, (200, 100, 50), 3)
    img = cv2.line(img, (25,0), (25, (width)), (200,100,69), 2)

    #cv2.imshow('Hand', img)
    return hand_in_cm


left_hand_x = point_x[2:5]
left_hand_y = point_y[2:5]
#print (left_hand_x, left_hand_y)
right_hand_x = point_x[5:8]
right_hand_y = point_y[5:8]

#print (left_hand_x, left_hand_y)

left_hand = hands_measurements(left_hand_x, left_hand_y, user_height, height)
right_hand = hands_measurements(right_hand_x, right_hand_y, user_height, height)
hand = (right_hand + left_hand)/2

print ('Lenght of left hand: ' + str(left_hand))
print ('Lenght of right hand: ' + str(right_hand))
print ('Lenght of the arm: ' + str(hand))

cv2.waitKey(0)

# Shoulders
def shoulder_length(shoulder_point_1, shoulder_point_2, body_size_cm, body_size_coordinates):
    shoulder_x = abs(int((shoulder_point_1[0]-shoulder_point_1[1]))) 
    shoulder_y = abs(int((shoulder_point_2[0]-shoulder_point_2[1])))
    shoulder_vector = (shoulder_x*shoulder_x) + (shoulder_y*shoulder_y)
    lenght_of_shoulder = math.sqrt(shoulder_vector)
#    print(lenght_of_shoulder)
    shoulder_in_cm = (lenght_of_shoulder * body_size_cm)/(body_size_coordinates)
    return shoulder_in_cm


shoulder_x = [point_x[2], point_x[5]]
shoulder_y = [point_y[2], point_y[5]]
shoulder = shoulder_length(shoulder_x, shoulder_y, user_height, height)
print('Length of the shoulder: ' + str(shoulder))

img = cv2.imread("mecd.jpg") # Image must be in the original size, because openpose did not include ROI

# Find head to make a proportion to the body for further calculations:
# todo: continue further body parts allocation using the proportion
face_cascade = cv2.CascadeClassifier('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/haarcascades/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

head = img[y:y+h, x:x+w]
height_head = head.shape[0]
width_head = head.shape[1]

head_length = (height_head * user_height)/height
print ('Head height: ' + str(head_length))

cv2.imshow('Head', head)
cv2.waitKey(0)
cv2.waitKey()
