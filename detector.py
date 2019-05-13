import numpy as np
import tensorflow as tf
import cv2 as cv
import load_inference_graph as setupgraph
import cvproc as proc
import argparse #argument passing
import collect_data as cold
import logging
import threading
import time
import socket
import queue
parser = argparse.ArgumentParser()
parser.add_argument(
    '--camerasrc',
    type = int,
    default=0,
    dest = 'camera_id')
parser.add_argument(
    '--RGB',
    type = bool,
    default=True,
    dest = 'imgcolor')
parser.add_argument(
    '--graphpath',
    type = str,
    default='frozen_inference_graph.pb',
    dest = 'graphpath')
parser.add_argument(
    '--score',
    type = float,
    default= 0.15, #change to 0.1-0.3 if there is objects in the background
    dest = 'score_thr')

args = parser.parse_args()
camera_id = args.camera_id
imgcolor = args.imgcolor
graphpath = args.graphpath
score_thr = args.score_thr
data_recording = True
#loading inference graph into memory
detector,session = setupgraph.load_inference_graph(graphpath)
print('\n Graph Locked and Loaded... \n')
print('Using Camera ID: ' + str(camera_id))

#capture input video stream
cap = cv.VideoCapture(camera_id)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

#width and height of frame
width  = cap.get(3)
height = cap.get(4)
#width and height of frame

#number of finger tips to detect
num_tips = 5

#cordinates of training data (features)
data_str = ""
gesture_number = "0,0,0,0,1\n"

#roi = frame[180:540, 180:540]

#check if camera was opened without errors
if(cap.isOpened() == False):
    print("Error: Video Capture Failed. \n Check Line 17 in detector.py\n")

keras_model = tf.keras.models.load_model('/home/hussam/Desktop/train_gesture/ges2/tmp/99/keras_trn2.model')
prediction = [1,0,0,0,0]
index = 0
count = 0


host = '67.20.211.228'
port  = 40123
size = 64
begin_sending = False
data = [135,135,135]
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((host,port))
buffer = queue.Queue(maxsize=100)

rec = True


#capture frames:
while(True):
    #capture frame by frame
    ret, frame = cap.read()
    #convert image to RGB
    if(imgcolor):
        try:
            #print('try')
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        except:
            print('Error: Could not convert to RGB. \n Check line 37 in detector.py \n')

    #get coordiantes of finger tips
    boxes, scores = setupgraph.detect_objects(frame,detector, session)
    finger_cord,left_most, right_most, top_most, bottom_most = proc.draw_box(num_tips, score_thr, scores, boxes, width, height, frame)
    center_of_hand = []
    cord_index = []

    if(len(finger_cord) == 5):
        center_of_hand = proc.center_of_hand(frame, finger_cord)
        cord_index = proc.return_cord_of_given_finger(finger_cord, 4, frame)

    key = cv.waitKey(33)
    if ((key) == 27):
        data = [-1,-2,-2]
        str1 = str(data[0]) + ',' + str(data[1]) + ',' + str(data[2])
        # s.send(str1.encode())
        break
    elif(key == 32):
        #cold.save_file(data_str, gesture_number)
        begin_sending = True
    else:
        if(len(cord_index) > 0 and len(center_of_hand) > 1):
            data_str, prediction, index, count, angs = cold.setup_file(gesture_number, width, height, cord_index, center_of_hand[0], center_of_hand[1], frame, keras_model, left_most, right_most, top_most, bottom_most, prediction, index, count)
            # if(begin_sending==True):
            #     data = angs
            #     if data[0] > 19 and data[0] < 136 and  data[1] > 19 and data[1] < 136 and  data[2] > 19 and data[2] < 136:
            #         str1 = str(data[0]) + ',' + str(data[1]) + ',' + str(data[2])
            #         buffer.put(str1)
            #         if rec:
            #             s.send(buffer.get().encode())
            #             xx = s.recv(64)
            #             if xx:
            #                 rec = True
            #             else:
            #                 rec = False

    upper_left = (int(width/4 - 36) , int(height/4 - 50))
    lower_right = (int(width*3/4 + 36) , int(height*3/4 + 50))
    cv.rectangle(frame, upper_left, lower_right , (255,0,0), 2)
    #cv.rectangle(frame, (140,80), (650,560), (255,0,0),0)
    cv.imshow('Capstone', cv.cvtColor(frame,cv.COLOR_RGB2BGR))

#Release the capture:
cap.release()
cv.destroyAllWindows()
