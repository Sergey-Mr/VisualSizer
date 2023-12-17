# cd Documents/PyProjects/Body-measurements/scripting-2/

import cv2
import imutils
import numpy as np
import os
import math
from statistics import mean
from roifind import roi_find
from json_data_out import get_points
#from find_head import find_head
from matplotlib import pyplot as plt

user_height = 176
mask = cv2.imread('output-2.png')

_, th1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # if pixel`s colour is more than 0 it transforms it in white

#plt.hist(mask.ravel(),256,[0,256]); plt.show()
#hist = cv2.calcHist([mask],[0],None,[256],[0,256])
white = [255,255,255]

# Get X and Y coordinates of all white pixels
Y, X = np.where(np.all(mask==white,axis=2))
X = X.tolist()
Y = Y.tolist()

#print (len(X), len(Y))

img = cv2.circle(mask, (int(X[0]), int(Y[0])), 3, (200, 100, 50), 3) # top point
# testing points
img = cv2.circle(mask, (89, 112), 3, (200, 100, 50), 3)
img = cv2.circle(mask, (98, 112), 3, (200, 100, 50), 3)
img = cv2.circle(mask, (173, 358), 3, (200, 100, 50), 3)
img = cv2.circle(mask, (91, 358), 3, (200, 100, 50), 3)
#cv2.imshow('Point', img)

# Divide body into parts
coef = user_height/7  # length of head
#print (coef)
height = mask.shape[0]
width = mask.shape[1]

head_length = (height*coef)/user_height  # length of head in pixels

# ROI
x_top = X[0]
y_top = Y[0]

y_bottom = Y[-1]
x_bottom = X[-1]

x_right = max(X)
x_ind = X.index(x_right)
y_right = Y[x_ind]

x_left = min(X)
x_ind_min = X.index(x_left)
y_left = Y[x_ind_min]

#print(x_top, y_top, x_bottom, y_bottom, x_right, y_right, x_left, y_left)
img = cv2.rectangle(img,(x_left,y_top),(x_right,y_bottom),(0,255,0),3)

head= int(y_top + head_length)
#img = cv2.rectangle(img, (x_left, y_top), (x_right, head), (0,255,240),3)
head_img = img[y_top:head, x_left:x_right]
cv2.imshow("Head", head_img)

chest = int((y_top + 2* head_length))
#img = cv2.rectangle(img, (x_left, y_top), (x_right, chest), (100,100,240),3)
chest_img = img[head:chest, x_left:x_right]
cv2.imshow("Chest", chest_img)

waist = int((y_top + 3* head_length))
#img = cv2.rectangle(img, (x_left, y_top), (x_right, waist), (100,120,140),3)
waist_img = img[chest:waist, x_left:x_right]
cv2.imshow("stomach", waist_img)

hips = int ((y_top + 4* head_length))
#img = cv2.rectangle(img, (x_left, y_top), (x_right, hips), (100,120,140),3)
hips_img = img[waist:hips, x_left:x_right]
cv2.imshow('Hips', hips_img)

# Contours
image = cv2.imread('output-2.png')
im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# In hips: exclude points next to beginning and the end
coord_index = []
x_exc_left = []
x_broke = []
y_broke = []
for i in range (len(X)):
    #print(i)
    if (X[i+1] - X[i])  < 10: 
        x_exc_left.append(i)

    else:
       #print (X[i], X[i+1])
       x_broke.append(X[i])
       x_broke.append(X[i+1])

       y_broke.append(Y[i])
       y_broke.append(Y[i+1])
       coord_index.append(i)
       
       break
#i = len(X)
#x_exc_right = []
#x_broke_r = []
#y_broke_r = []
#coord_index_r = []
#while i != 0:
#    if (X[i-1] - X[i-2]) < 10:
#        x_exc_right.append(i)
#    else:
#        print (X[i-1], X[i-2])
#        x_broke_r.append(X[i-1])
#        x_broke_r.append(X[i-2])
#
#        y_broke_r.append(Y[i-1])
#        y_broke_r.append(Y[i-2])
#        coord_index_r.append(i)
#        break
#    i-=1
#print ('Hips', x_broke, y_broke)
#print ('Hips_r', x_broke_r, y_broke_r)

# Left hand exclusion
no_interest = image[0:max(Y), 0:x_broke[1]] # roi
cv2.imshow("no_interest", no_interest)
distance = x_broke[1] # distance between the beginning of the image and the beginning of the right hand
left_part_hand_start = max(X) - distance

roi = image[(y_broke[1]-10):(y_broke[1]+10), x_broke[1]:max(X)]
black_pixels = np.where(roi == 0)
if len(black_pixels[1]) > 0:
    first_black_pixel = (black_pixels[0][np.min(black_pixels[0])], np.min(black_pixels[1]))
    print(f"First black pixel coordinates: ({first_black_pixel[1]}, {first_black_pixel[0]})")
else:
    print("No black pixels found in the no_interest_left array.")
cv2.circle(roi, (68,0), 3, (0, 255, 0), 2)

cv2.circle(image, (x_broke[1], y_broke[1]), 3, (0, 0, 255), 2)
cv2.circle(image, (left_part_hand_start, y_broke[1]), 3, (0, 0, 255), 2)
no_interest_left = image[0:max(Y), left_part_hand_start:max(X)]
chest_length_px = int(left_part_hand_start - first_black_pixel[1])
chest_length_cm = (chest_length_px * user_height)/height

print("Chest length:", chest_length_cm)
cv2.imshow("Final", roi)

#hips_img = img[waist:hips, x_left:x_right]
#hips_img_not_r = img[waist:hips, 0:x_broke[1]]
#hips_img_not_l = img[waist:hips, left_part_hand_start:max(X)]
#cv2.imshow('HIPS', hips_img_not_l)

#points = get_points('mecd_keypoints.json')
#i = 0
#points_group = []
#while i <= len(points) - 3:
#    a = [points[i],points[i+1],points[i+2]]
#    points_group.append(a)
#    i += 3
#
#print (points_group)
## Finding xs and ys
#point_x = []
#point_y = []
#for point in points_group:
#    for x in point:
#        if x == point[0]:
#           point_x.append(x)
#        elif x == point[1]:
#            point_y.append(x)
#        else:
#            pass



#plt.imshow(th1)
#plt.show()
cv2.imshow('Mask', img)
cv2.waitKey(0)
