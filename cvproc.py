#frame processing script
import math
import tensorflow as tf
import cv2 as cv
import numpy as np

def draw_box(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    finger_cord = []
    left_most = [720,0]
    right_most = [0,0]
    top_most = [720,720]
    bottom_most = [0,0]
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            if(left<left_most[0]):
                left_most = [left,top]
            if(right>right_most[0]):
                right_most = [right,top]
            if(top < top_most[1]):
                top_most = [left,top]
            if(bottom > bottom_most[1]):
                bottom_most = [left, top]
            temp = [left , right, top, bottom, scores[i]]
            finger_cord.append(temp)
            cv.rectangle(image_np, p1, p2, (255, 255, 9), 3, 1)
    return sorter(finger_cord), left_most, right_most, top_most, bottom_most

def sorter(arr):
    arr.sort(key=lambda x:x[0])
    return arr


def center_of_hand(img,finger):
    # cord = []
    # cord_x = (finger[0][0] + finger[4][1]) / 2
    # cord_y = (finger[0][3] + finger[4][3]) - 300
    # cv.circle(img, (int(cord_x),int(cord_y) ), 10, (255,0,0) , -1)
    cord = [1, 1]
    return cord



def return_cord_of_given_finger(finger,id, img):
    cord = []
    if (id == 1):
        #pinky
        id = 1
    elif(id == 2):
        #ring finger
        id =2
    elif(id == 3):
        #middle finger
        id =3
    elif(id == 4):
        #index finger
        centerx = (finger[3][0] + finger[3][1]) / 2
        centery = (finger[3][2] + finger[3][3]) / 2
        height = abs(finger[3][2] - finger[3][3])
        width = abs(finger[3][1] - finger[3][0])
        score = finger[3][4]
        cord = [centerx, centery, width, height,score]
        cv.circle(img, (int(centerx),int(centery) ), 5, (255,255,255), -1)
    else:
        #thumb
        id =4
    return cord
