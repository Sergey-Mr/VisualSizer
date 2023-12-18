import cv2
import imutils
import numpy as np
import os
import math
from roifind import roi_find
from json_data_out import get_points
from mask_research import mask_research_func
from pytorch_working import mask_create

def three_point_length (points_x, points_y, body_size_cm, body_size_coordiantes, **img): 
    '''Finds the length of the 3-point body part using the openpose points'''
    
    # Formula: sqrt ((x1-x0)^2 + (y1-y0)^2)
    # First vector
    length_v_x_1 = (int(points_x[0])- int(points_x[1]))**2
    length_v_y_1 = (int(points_y[0])- int(points_y[1]))**2
    length_v_1 = abs(length_v_x_1 + length_v_y_1)
    length_v_1 = math.sqrt(length_v_1)
    #print (length_v_1)

    # Second vector
    length_v_x_2 = (int(points_x[2])- int(points_x[1]))**2
    length_v_y_2 = (int(points_y[2])- int(points_y[1]))**2
    length_v_2 = abs(length_v_x_2 + length_v_y_2)
    length_v_2 = math.sqrt(length_v_2)
    #print (length_v_2)

    # Mean 
    lenght_of_hand = length_v_1 + length_v_2
    final_in_cm = (lenght_of_hand * body_size_cm)/(body_size_coordiantes)

    return final_in_cm


def two_point_length(points_x, points_y, body_size_cm, body_size_coordinates):
    '''Find the length of the 2-point body part in cm using openpose points'''
    shoulder_x = abs(int((points_x[0]-points_x[1]))) 
    shoulder_y = abs(int((points_y[0]-points_y[1])))
    shoulder_vector = (shoulder_x*shoulder_x) + (shoulder_y*shoulder_y)
    lenght_of_shoulder = math.sqrt(shoulder_vector)
    final_in_cm = (lenght_of_shoulder * body_size_cm)/(body_size_coordinates)
    return final_in_cm


def run_openpose(openpose_location, net_resolution, image_dir, json_out, image_out):
    '''Start the openpose script'''
    current_directory = os.getcwd()
    os.chdir(openpose_location)
    print(os.getcwd())
    os.system('./build/examples/openpose/openpose.bin --net_resolution -'+ net_resolution +' --image_dir ' + image_dir + ' --write_json ' + json_out +' --write_images ' + image_out)
    os.chdir(current_directory)
    print(os.getcwd())


def detect_edges(input_image):
    '''User derivatives to find the contour of the body'''
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    input_image = cv2.imread(input_image)
    grad_x = cv2.Sobel(input_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(input_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    print(grad_y)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow('Gradients', grad)



image_name = "mecd.jpg"
openpose_location = '/home/serhii/openpose'
net_resolution = '1x128'
image_dir = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement/images/'
json_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement'
image_out = '/home/serhii/Documents/PyProjects/Body-measurements/main-git/Body-measurement/images'
haarcascades_path = 'haarcascades/haarcascade_frontalface_default.xml'

image = cv2.imread(image_name)

# ROI
x, y, w, h = roi_find(image_name)

#user_height = int(input('Input your height:')) # y+h
user_height = 177

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
#run_openpose(openpose_location, net_resolution, image_dir, json_out, image_out)

# Getting points from openpose
points = get_points('mecd_keypoints.json')

i = 0
points_group = []
while i <= len(points) - 3:
    a = [points[i],points[i+1],points[i+2]]
    points_group.append(a)
    i += 3

# Separate into Xs and Ys
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
left_hand = three_point_length(left_hand_x, left_hand_y, user_height, height)
right_hand = three_point_length(right_hand_x, right_hand_y, user_height, height)
hand = (right_hand + left_hand)/2

#print ('Length of left hand: ' + str(left_hand))
#print ('Length of right hand: ' + str(right_hand))
print ('Length of the arms: ' + str(hand))

# Calculate leg's length in cm
left_leg_x = point_x[9:12]
left_leg_y = point_y[9:12]
right_leg_x = point_x[12:15]
right_leg_y = point_y[12:15]

left_leg = three_point_length(left_leg_x, left_leg_y, user_height, height)
right_leg = three_point_length(right_leg_x, right_leg_y, user_height, height)
print('Length of the legs: ', (left_leg+right_leg)/2)

# Calculate the length of shoulders
shoulder_x = [point_x[2], point_x[5]]
shoulder_y = [point_y[2], point_y[5]]
shoulder = two_point_length(shoulder_x, shoulder_y, user_height, height)
print('Length of the shoulder: ' + str(shoulder))

# Calculate the lenght of torse
torse_x = [point_x[1], point_x[8]]
torse_y = [point_y[1], point_y[8]]
torse = two_point_length(torse_x, torse_y, user_height, height)
print('Length of the torse: ' + str(torse))

img = cv2.imread(image_name) # Image must be in the original size, because openpose did not include ROI

# Detect face
face_cascade = cv2.CascadeClassifier(haarcascades_path)
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
#mask_create(image_name)
head_img, chest_img, waist_img, hips_img = mask_research_func('output.png')
detect_edges('output.png')

cv2.imshow('Head-100', head_img)
cv2.waitKey(0)