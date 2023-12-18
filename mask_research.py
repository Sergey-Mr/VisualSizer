import cv2
import imutils
import numpy as np
import os
import math
from matplotlib import pyplot as plt

#image_name = 'output.png'

def mask_research_func(image_name): 
    user_height = 176
    mask = cv2.imread(image_name)

    # White mask application 
    _, th1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # if pixel`s colour is more than 0 it transforms it in white
    white = [255,255,255]

    # Get X and Y coordinates of all white pixels
    Y, X = np.where(np.all(mask==white,axis=2))
    X = X.tolist()
    Y = Y.tolist()

    img = cv2.circle(mask, (int(X[0]), int(Y[0])), 3, (200, 100, 50), 3) # top point

    # testing points
    #img = cv2.circle(mask, (89, 112), 3, (200, 100, 50), 3)
    #img = cv2.circle(mask, (98, 112), 3, (200, 100, 50), 3)
    #img = cv2.circle(mask, (173, 358), 3, (200, 100, 50), 3)
    #img = cv2.circle(mask, (91, 358), 3, (200, 100, 50), 3)
    #cv2.imshow('Point', img)

    # Divide body into parts
    coef = user_height/7  # length of head
    height = mask.shape[0]
    width = mask.shape[1]

    head_length = (height*coef)/user_height  # length of head in pixels according to the proportion data

    # ROIs
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

    #img = cv2.rectangle(img,(x_left,y_top),(x_right,y_bottom),(0,255,0),3)

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

    image = cv2.imread(image_name)

    # Find white pixels' distance difference 
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

    # Left hand exclusion
    no_interest = image[0:max(Y), 0:x_broke[1]] # roi
    cv2.imshow("no_interest", no_interest)

    distance = x_broke[1] # distance between the beginning of the image and the beginning of the right hand
    left_part_hand_start = max(X) - distance

    roi = image[(y_broke[1]-10):(y_broke[1]+10), x_broke[1]:max(X)]
    black_pixels = np.where(roi == 0)

    # Find the first black pixel on the RHS representing the start of the hand
    if len(black_pixels[1]) > 0:
        first_black_pixel = (black_pixels[0][np.min(black_pixels[0])], np.min(black_pixels[1]))
        print(f"First black pixel coordinates: ({first_black_pixel[1]}, {first_black_pixel[0]})")

    else:
        print("No black pixels found in the no_interest_left array.")

    cv2.circle(roi, (68,0), 3, (0, 255, 0), 2)
    cv2.circle(image, (x_broke[1], y_broke[1]), 3, (255, 0, 0), 2)

    #new_roi = img[y_top:head, x_broke[1]:first_black_pixel[1]]
    #hips_img = img[waist:hips, x_left:x_right]
    #cv2.imshow('NEW ROI', hips_img)

    #no_interest_left = image[waist:hips, x_broke[1]:black_pixels[1]]

    chest_length_px = int(left_part_hand_start - first_black_pixel[1])
    chest_length_cm = (chest_length_px * user_height)/height

    print("Chest length:", chest_length_cm)
    #cv2.imshow("Final", roi)
    #cv2.imshow('Result', image)
    #cv2.imshow('Mask', img)

    #cv2.waitKey(0)
    return head_img, chest_img, waist_img, hips_img


#mask_research(image_name)