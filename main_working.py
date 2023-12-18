import cv2
import imutils
import numpy as np
import os
import math
from roifind import roi_find
from json_data_out import get_points

def hands_measurements (hand_points_x, hand_points_y, body_size_cm, body_size_coordiantes, **img): 
    '''Finds the length of the amr using the openpose points'''
    
    # sqrt ((x1-x0)^2 + (y1-y0)^2)
    # Left hand
    length_v_x_1 = (int(hand_points_x[0])- int(hand_points_x[1]))**2
    length_v_y_1 = (int(hand_points_y[0])- int(hand_points_y[1]))**2
    length_v_1 = abs(length_v_x_1 + length_v_y_1)
    length_v_1 = math.sqrt(length_v_1)
    #print (length_v_1)

    # Right hand
    length_v_x_2 = (int(hand_points_x[2])- int(hand_points_x[1]))**2
    length_v_y_2 = (int(hand_points_y[2])- int(hand_points_y[1]))**2
    length_v_2 = abs(length_v_x_2 + length_v_y_2)
    length_v_2 = math.sqrt(length_v_2)
    #print (length_v_2)

    # Mean 
    lenght_of_hand = length_v_1 + length_v_2
    hand_in_cm = (lenght_of_hand * body_size_cm)/(body_size_coordiantes)

    return hand_in_cm


def shoulder_length(shoulder_point_1, shoulder_point_2, body_size_cm, body_size_coordinates):
    '''Find the length of the shoulders in cm using openpose points'''
    shoulder_x = abs(int((shoulder_point_1[0]-shoulder_point_1[1]))) 
    shoulder_y = abs(int((shoulder_point_2[0]-shoulder_point_2[1])))
    shoulder_vector = (shoulder_x*shoulder_x) + (shoulder_y*shoulder_y)
    lenght_of_shoulder = math.sqrt(shoulder_vector)
    shoulder_in_cm = (lenght_of_shoulder * body_size_cm)/(body_size_coordinates)
    return shoulder_in_cm


def run_openpose(image_dir, json_out, image_out):
    os.chdir('/home/serhii/openpose')
    print(os.getcwd())
    os.system('./build/examples/openpose/openpose.bin --net_resolution -1x128 --image_dir ' + image_dir + ' --write_json ' + json_out +' --write_images ' + image_out)
    os.chdir('/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement')
    print(os.getcwd())


image_name = "mecd.jpg"
image_dir = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement/images/'
json_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement'
image_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement/images'

image = cv2.imread(image_name)

# ROI
x, y, w, h = roi_find(image_name)

#user_height = int(input('Input your height:')) # y+h
user_height = 180

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

# ROI
image = image[y:y+h, x:x+w]
height = image.shape[0]
width = image.shape[1]

#Run openpose
run_openpose(image_dir, json_out, image_out)

# Getting points from openpose
points = get_points('mecd_keypoints.json')
i = 0
points_group = []
while i <= len(points) - 3:
    a = [points[i],points[i+1],points[i+2]]
    points_group.append(a)
    i += 3

# Separate into xs and ys
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

# Define points of arms
left_hand_x = point_x[2:5]
left_hand_y = point_y[2:5]
right_hand_x = point_x[5:8]
right_hand_y = point_y[5:8]

# Calculate arm's length in cm
left_hand = hands_measurements(left_hand_x, left_hand_y, user_height, height)
right_hand = hands_measurements(right_hand_x, right_hand_y, user_height, height)
hand = (right_hand + left_hand)/2

print ('Length of left hand: ' + str(left_hand))
print ('Length of right hand: ' + str(right_hand))
print ('Length of the arm: ' + str(hand))

# Calculate leg's length in cm
left_leg_x = point_x[9:12]
left_leg_y = point_y[9:12]
right_leg_x = point_x[12:15]
right_leg_y = point_y[12:15]

left_leg = hands_measurements(left_leg_x, left_leg_y, user_height, height)
right_leg = hands_measurements(right_leg_x, right_leg_y, user_height, height)

# Calculate the length of shoulders
shoulder_x = [point_x[2], point_x[5]]
shoulder_y = [point_y[2], point_y[5]]
shoulder = shoulder_length(shoulder_x, shoulder_y, user_height, height)
print('Length of the shoulder: ' + str(shoulder))

# Calculate the lenght of torse
torse_x = [point_x[1], point_x[8]]
torse_y = [point_y[1], point_y[8]]
torse = shoulder_length(torse_x, torse_y, user_height, height)
print('Length of the torse: ' + str(torse))

img = cv2.imread(image_name) # Image must be in the original size, because openpose did not include ROI

# Detect face
face_cascade = cv2.CascadeClassifier('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/haarcascades/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

head = img[y:y+h, x:x+w]
height_head = head.shape[0]
width_head = head.shape[1]

head_length = (height_head * user_height)/height
print ('Head length: ' + str(head_length))

#cv2.imshow('Head', head)
#use the proportion for the further calculation, but first integrate pytorch thing

#import pytorch_working_01  # Create a black-white mask using pytorch
import mask_research

cv2.waitKey(0)