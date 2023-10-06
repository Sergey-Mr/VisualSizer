import cv2
import numpy as np

def nothing(x):
    pass


#  HSV (Hue, Saturation, Value)
#  for detecting lower and upper datas of the color you should use Tackle Bars
cv2.namedWindow('Tracking')

cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing)  # lower hue
cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)  # lower saturation
cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)  # lower value
cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing)  # upper hue
cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)  # upper saturation
cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)  # upper value

while True:
    frame = cv2.imread('fallowfield-1.jpg')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converting image in HSV
    #  vars, that contains got datas form the track bars
    l_h = cv2.getTrackbarPos('LH', 'Tracking')
    l_s = cv2.getTrackbarPos('LS', 'Tracking')
    l_v = cv2.getTrackbarPos('LV', 'Tracking')

    u_h = cv2.getTrackbarPos('UH', 'Tracking')
    u_s = cv2.getTrackbarPos('US', 'Tracking')
    u_v = cv2.getTrackbarPos('UV', 'Tracking')

    l_b = np.array([l_h, l_s, l_v])  # lower blue color
    u_b = np.array([u_h, u_s, u_v])  # upper blue color

    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask=mask)  # bitwise operation for mask

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()