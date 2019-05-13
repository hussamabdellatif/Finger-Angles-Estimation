import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)

while(1):

    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # range of the skin colour is defined
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
     #extract skin colur image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
    #image is blurred using GBlur
        mask = cv2.GaussianBlur(mask,(5,5),100)
    #find contours
        _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)

        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
    k = cv2.waitKey(33)
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
