# cd Documents/PyProjects/Body-measurements/scripting-2/

import cv2
import imutils
import numpy as np
import os
import math
from statistics import mean
from roifind import roi_find
from json_data_out import get_points
from find_head import find_head
from matplotlib import pyplot as plt

user_height = 176
mask = cv2.imread('output.png')

_, th1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # if pixel`s colour is more than 0 it transforms it in white

#plt.hist(mask.ravel(),256,[0,256]); plt.show()
#hist = cv2.calcHist([mask],[0],None,[256],[0,256])
white = [255,255,255]

# Get X and Y coordinates of all black pixels
Y, X = np.where(np.all(mask==white,axis=2))
X = X.tolist()
Y = Y.tolist()

print (len(X), len(Y))

img = cv2.circle(mask, (int(X[0]), int(Y[0])), 3, (200, 100, 50), 3)

#Find head 1
'''
head = []
head_upper = []
head_upper_y = []
head_bottom_x = []  
head_bottom_y = []
i=0
while i < (len(X)-1):
    x_i = X[i]
    x_i1 = X[i+1]
#    print (x_i, x_i1)

    y_i = Y[i]
    y_i1 = Y[i+1]
#    print (y_i, y_i1)

    if y_i<y_i1 and x_i > x_i1:
        print ('Ys: ' + str(y_i) + ' < ' + str (y_i1))
        print('Xs: ' + str(x_i) + ' > ' + str (x_i1))
        head_upper.append([x_i, y_i])
    else: 
        print ('Nothing found')
    i+=1

print (head_upper)
'''

# Find head 2
'''
height = mask.shape[0]
width = mask.shape[1]
image = cv2.imread('mecd.jpg')
image = cv2.resize(image, (width, height))
cv2.imshow('reshaped', image)

face_cascade = cv2.CascadeClassifier('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/haarcascades/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

print (y+h)
print (height)
head = mask[y:y+h, x:x+w]
height_head = head.shape[0]
width_head = head.shape[1]

head_length = (height_head * 176)/height
print ('Head height: ' + str(head_length))
cv2.imshow ('Point', img)
#print (X, Y)
'''

'''GET COORDINATES OF WHITE PIXELS and research their position from top to bottom'''
"Make head detection using it"

# Divide body into parts
coef = user_height/7  # length of head
print (coef)
height = mask.shape[0]
width = mask.shape[1]

head_length = (height*coef)/user_height  # length of head in pixels
print(head_length)

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

print(x_top, y_top, x_bottom, y_bottom, x_right, y_right, x_left, y_left)
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
image = cv2.imread('output.png')
im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#(thresh, im_bw) = cv2.threshold(im_bw, 0, 255, 0)
#contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#print (contours)

#contours_list = []
#for contour in contours:
#    if cv2.contourArea(contour) < 3000:  # for removing noises
#        continue
#    cv2.drawContours(image,[contour],-1, (0, 255, 0), 3)

#cv2.imshow("Contours", image)


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
       print (X[i], X[i+1])
       x_broke.append(X[i])
       x_broke.append(X[i+1])

       y_broke.append(Y[i])
       y_broke.append(Y[i+1])
       coord_index.append(i)
       
       break
       
print (coord_index)

# Следующий кардинаты по прямой нужно убрать из ROI, если они чёрные

print (max(X))

ROI = mask[chest:waist, x_broke[1]:max(X)]
cv2.line(image, (x_broke[1], y_broke[1]), (max(X), y_broke[1]), (140, 20, 60), 2)
cv2.imshow('ROI', ROI)
'''we have a steady decrease of x and y of the bottom part of the arm from the LHS'''

#y_roi, x_roi = np.where(np.all(mask==white,axis=2))
#x_roi = x_roi.tolist()
#y_roi = y_roi.tolist()

#for i in range(len(x_roi)):
#    print(i)

'''Найти координаты руки, провести линию и найти ближаюшую белую точку. Удалить расстояние между ними '''
'''Если игрек пикселя  увеличивался, а потом начал резко уменьшатсь'''

y_right_hand = Y[X.index(max(X))]
right_hand_c = [max(X), y_right_hand]
print (right_hand_c, y_right_hand)
cv2.circle(image, (int(right_hand_c[0]), int(right_hand_c[1])), 2, (0, 0, 255), 3)

y_left_hand = Y[X.index(min(X))]
left_hand_c = [min(X), y_left_hand]
print (left_hand_c, y_left_hand)
cv2.circle(image, (int(left_hand_c[0]), int(left_hand_c[1])), 2, (0, 0, 255), 3)
# print(X)

# ROI (using openpose). Indexes: 2 - left shoulder, 5 - right shoulder
# Getting points from openpose
mecd = cv2.imread('mecd.jpg')
print('HERE')
print (mecd.shape[0], mecd.shape[1])
print (mask.shape[0], mask.shape[1])

coefficient = ((mecd.shape[0]/mask.shape[0]) + (mecd.shape[1]/mask.shape[1]))/2
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

image_original = cv2.imread('mecd.jpg')
image_original = cv2.resize(image_original, (width, height))
cv2.imwrite('mecd_resized.jpg', image_original)


#Run openpose
#os.chdir('/home/serhii/openpose')
#print(os.getcwd())
#os.system('./build/examples/openpose/openpose.bin --net_resolution -1x128 --image_dir /home/serhii/Documents/PyProjects/Body-measurements/scripting-2/resized_image --write_json /home/serhii/Documents/PyProjects/Body-measurements/scripting-2/ --write_images /home/serhii/Documents/PyProjects/Body-measurements/scripting-2/')
#os.chdir('/home/serhii/Documents/PyProjects/Body-measurements/scripting-2/')
#print(os.getcwd())

# Getting points from openpose
points = get_points('mecd_resized_keypoints.json')
i = 0
points_rersized_group = []
while i <= len(points) - 3:
    a = [points[i],points[i+1],points[i+2]]
    points_rersized_group.append(a)
    i += 3

print(points_rersized_group)

point_x_resized = []
point_y_resized = []
for point in points_rersized_group:
    for x in point:
        if x == point[0]:
           point_x_resized.append(x)
        elif x == point[1]:
            point_y_resized.append(x)
        else:
            pass
print(point_x_resized)
print (point_y_resized)

print('Resized points')
print (point_x_resized[2], point_x_resized[5])

cv2.circle(image, (int(point_x_resized[2]), y_right_hand), 2, (0, 0, 255), 3)
cv2.circle(image, (int(point_x_resized[5]), y_left_hand), 2, (0, 0, 255), 3)

roi = cv2.imread('output.png')
#roi_no_hands = roi[0:max(Y),int(point_x_resized[2]-10):int(point_x_resized[5])+10]  # +10 is approximation 

# Сделать + 10 и - 10 и начать с самой верхней точки

half = int(max(X))/2

print(int(point_x_resized[2])-10)
roi_no_hands_left = roi[0:max(Y),int(point_x_resized[2]-10):int(half)]
roi_no_hands_left = cv2.rotate(roi_no_hands_left, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Roi no hands', roi_no_hands_left)
#cv2.imwrite('ROI_no_hands_left.png', roi_no_hands_left)

''' 
Coefficients method
left_roi = [int(point_x[2])/coefficient, y_right_hand]
right_roi = [int(point_x[5])/coefficient, y_left_hand]

print('Coefficients')
print (left_roi[0], right_roi[0])
#cv2.circle(image, (int(left_roi[0]), int(left_roi[1])), 2, (0, 0, 255), 3)
#cv2.circle(image, (int(right_roi[0]), int(right_roi[1])), 2, (0, 0, 255), 3)
'''


# Make a list of points x for ROI
#roi_points = []y_right_hand
#for j in range(x_broke[1], x_right):
#    roi_points.append(j)
#
#print (roi_points)
#
#print ('Y')
#index_broke = int(coord_index[0])

# STOPPED HERE. REMEMBER THAT ROI (0;MAX), BUT J IS MORE THAN MAX. 
#for j in range (int(x_broke[0]), max(X)):
#    print (j)
#
#    y = int(Y[index_broke])
#    print (ROI[int(j-x_broke[1]), y])
    #print (b,g,r)
    #if b == 0 and g == 0 and r == 0:
    #    cv2.circle(ROI, (X[j-107], Y[int(coord_index[0])]), 5, (100, 230, 150), 3)
    #    break
    #else:
    #    continue


cv2.imshow("Contours, black", image)

img_zoomed = cv2.imread ('output.png')
cv2.circle(img_zoomed, (x_broke[0], y_broke[0]), 2, (200, 50, 30), 3)
cv2.circle(img_zoomed, (x_broke[1], y_broke[1]), 2, (100, 200, 50), 3)
img_zoomed = img_zoomed[(int(y_broke[0]-40)):(int(y_broke[1]+50)), (int(x_broke[0]-50)):(int(x_broke[1]+50))]

cv2.imshow('Zoomed', img_zoomed)

# Show where it broke on each image. TRY NOT TO USE X_EXC_LEFT, BUT X_BROKE[-1] INSTEAD  
cv2.circle(img, (X[x_exc_left[-1]], Y[x_exc_left[-1]]), 3, (100, 230, 150), 3)
cv2.circle(waist_img, (X[x_exc_left[-1]], Y[x_exc_left[-1]]), 3, (100, 230, 150), 3)
cv2.circle(hips_img, (X[x_exc_left[-1]], Y[x_exc_left[-1]]), 3, (100, 230, 150), 3)
cv2.circle(chest_img, (X[x_exc_left[-1]], Y[x_exc_left[-1]]), 3, (100, 230, 150), 3)

cv2.imshow('Hands_p1', chest_img)
cv2.imshow('Hands_p2', waist_img)
cv2.imshow('Hands_p3', hips_img)

'''
To Do: 
1. Come up with the correct method of measuring parts

Done:
1. Finished segmentation of the body into 4 ROIs based on ratio of length of head to body (1:7)
Length of head is measured by a proportion and recalculated for pixels based on user`s input

Possible solution:
1. Turn contour of the RHS and LHS bodies into line and represent as a function. 
2. Make solutions for different body types with if, else
3. Make a research based on derivative of the function

Git Hub repository: https://github.com/Stephanehk/Image_Based_Graph_Line_Equation_Extracting

'''


#plt.imshow(th1)
#plt.show()
cv2.imshow('Mask', img)
cv2.waitKey(0)

