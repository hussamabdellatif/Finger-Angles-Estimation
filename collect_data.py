import math
import cv2 as cv
import numpy as np
import threading

def setup_file(gesture_number, frame_width, frame_height, box, center_of_hand_x, center_of_hand_y, frame, model, left_most, right_most, top_most, bottom_most, prediction_result, index, count):
        #box is an array of [centerx, centery, w, h, score]
        if( len(box) != 0 and frame_width != 0 and frame_height != 0 and center_of_hand_x != 0 and center_of_hand_y != 0):
            #center_of_hand_x, center_of_hand_y, center_of_hand_h, center_of_hand_w = find_center_of_hand(frame)
            focal_length = math.pow(4.42, -3) #assume focal_length of 4.42 mm
            real_hight_of_object = math.pow(2,-2)
            camera_frame_height = frame_height
            image_height = box[3]
            sensor_height = math.pow(3.0 , -3)
            cord1 = math.sqrt( math.pow(box[0],2) + math.pow(box[1], 2)) #center of box to upper-left corner (0,0)
            cord2 = math.sqrt( math.pow(box[0],2) + math.pow(box[1] - frame_height , 2) ) #center of box to lower-left corner (0,h)
            cord3 = math.sqrt( math.pow(box[0] - frame_width ,2) + math.pow(box[1],2) ) #center of box to upper-right corner (w,0)
            cord4 = math.sqrt( math.pow(box[0] - frame_width, 2) + math.pow(box[1] - frame_height, 2) ) #center of box to lower-right corner (w,h)
            cord5 = box[0] #x axis coordiantes of center of box
            cord6 = box[1] #y axis coordinates of center of box
            cord7 = box[2] #w of box
            cord8 = box[3] #h of box
            score = box[4] #score of box
            cord9 = math.sqrt( math.pow(box[0] - (frame_width/2) , 2) + math.pow(box[1] - (frame_height/2),2)) #center of box to center of frame
            cord10 = math.sqrt( math.pow(box[0] - (frame_width/2) ,  2) + math.pow(box[1], 2)) #center of box to center top
            cord11 = math.sqrt( math.pow(box[0] - (frame_width/2) ,  2) + math.pow(box[1]-frame_height, 2)) #center of box to center bottm
            cord12 = ((focal_length * real_hight_of_object * camera_frame_height ) / (image_height*sensor_height) ) * 10 # distance of finger tip to camera
            cord13 = math.sqrt( math.pow(box[0],  2) + math.pow(box[1]-(frame_height/2), 2)) #center of box to center left
            cord14 = math.sqrt( math.pow(box[0] - frame_width ,  2) + math.pow(box[1]-(frame_height/2), 2)) #center of box to center right
            cord15 = math.sqrt(math.pow(left_most[0],2) + math.pow(left_most[1] - (frame_height/2) ,2)) #left_most to left_center of screen
            cord16 = math.sqrt(math.pow(right_most[0],2) + math.pow(right_most[1] - (frame_height/2) ,2)) #right_most to left_center of screen
            cord17 = math.sqrt(math.pow(top_most[0]-(frame_width/2),2) + math.pow(top_most[1] ,2)) #left_most to left_center of screen
            cord18 = math.sqrt(math.pow(bottom_most[0]-(frame_width/2),2) + math.pow(bottom_most[1] ,2)) #left_most to left_center of screen
            #cord15 = ((focal_length*math.pow(30,-3)*camera_frame_height) / (center_of_hand_h*sensor_height)) *10000 # distance of the center of the hand to the camera
            #cord16 = math.sqrt(math.pow(center_of_hand_x,2) + math.pow(center_of_hand_y-(frame_height/2) , 2)) #distance of the center of the hand to the left cener of the frame
            #cord17 = math.sqrt(math.pow(center_of_hand_x-frame_width,2) + math.pow(center_of_hand_y-(frame_height/2) , 2)) #distance of the center of the hand to the right cener of the frame
            #cord18 = math.sqrt(math.pow(center_of_hand_x - (frame_width/2),2) + math.pow(center_of_hand_y, 2)) #distance of the center of the hand to the top cener of the frame
            #cord19 = center_of_hand_x
            #cord20 = center_of_hand_y
            #cord21 = center_of_hand_h
            #cord22 = center_of_hand_w

            data_arr = [score, cord1, cord2, cord3, cord4, cord5, cord6, cord7, cord8, cord9, cord10, cord11, cord12, cord13, cord14, cord15, cord16, cord17, cord18]


            x_predict = np.asarray(data_arr, dtype=np.float32)
            x_predict = x_predict.reshape((1,19))
            prediction = model.predict(x_predict)
            max = -1
            pred_formatted = ['%.1f' % elem for elem in prediction[0]]
            max_index = 0
            for i in range(len(pred_formatted)):
                pred_formatted[i] = float(pred_formatted[i]) * 100
                if pred_formatted[i] > max:
                    max = pred_formatted[i]
            if max >= 90:
                for i in range(len(pred_formatted)):
                    if pred_formatted[i] == max:
                        if i != index:
                            max_index = i
                            count = count+1
                if count == 2 and max_index != index:
                    prediction_result = [0,0,0,0,0]
                    prediction_result[max_index] = 1
                    index = max_index
                    count = 0



            angs = find_angle(index, int(cord10))
            print('Angles: servo_one, servo_two, servo_three \n')
            print(angs)
            print('\n')
            print('Gesture Prediction')
            print(prediction_result)
            print("\n")
            data_str = str(score) + ',' + str(cord1) + ',' + str(cord2) + ',' + str(cord3) + ',' + str(cord4) + ',' + str(cord5) + ',' + str(cord6) + ',' + str(cord7) + ',' + str(cord8) + ',' + str(cord9) + ',' + str(cord10) +',' + str(cord11)+','+str(cord12)+',' + str(cord13)+',' + str(cord14)+',' + str(cord15)+',' + str(cord16)+',' + str(cord17)+',' + str(cord18) + "\n"
            cv.line(frame, (0,0), (int(box[0]), int(box[1])) , (255,0,0), 5)
            cv.line(frame, (0,int(frame_height) ), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (int(frame_width),0), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (int(frame_width),int(frame_height)), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (int(frame_width/2),0), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (int(frame_width/2),int(frame_height)), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (0,int(frame_height/2)), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            cv.line(frame, (int(frame_width),int(frame_height/2)), (int(box[0]), int(box[1])), (255,0,0 ), 5 )
            font = cv.FONT_HERSHEY_SIMPLEX
            str1 = "UL: " + str(int(cord1)) + ", LL: " + str(int(cord2)) + ", UR: " + str(int(cord3)) + ", LR: " + str(int(cord4)) + ", CL: " + str(int(cord13)) + ", CR: " + str(int(cord14))
            str2 = "Score: " + str(score) + " C_to_CoF: " + str(int(cord9)) + ", C_to_TC: " + str(int(cord10))+ ", C_to_BC: "+str(int(cord11))  + ", DtoCam: " + str(int(cord12))
            #str3 = "center_of_hand: " + str(cord15) + " COH_to_lefth " + str(int(cord16)) + " COH_to_toph " + str(int(cord18)) + " coh_x: " + str(int(cord19)) + " coh_y: " + str(int(cord20)) + " coh_w: " + str(int(cord22))
            cv.putText(frame, str1, (2, 430), font, 0.5, (0,255,0))
            cv.putText(frame, str2, (2, 470), font, 0.5, (0,255,0))
            return  data_str, prediction_result, index, count, angs


def find_angle(index, distance):
    servo_one = 0
    servo_two = 0
    servo_three = 0
    if(index == 0):
        servo_three = 135
        servo_two = 135
        servo_one = int((-23/35)*distance + (2210/7))
    if(index == 1):
        servo_one = 135
        servo_three = 135
        servo_two = int((-23/24)*distance + (4265/12))
    if(index == 2):
        servo_one = 50
        servo_three = 135
        servo_two = int((-23/28)*distance + (480))
    if(index == 3):
        servo_one = 135
        servo_two = 30
        servo_three = 40
    if(index ==4):
        servo_one = 135
        servo_two = 135
        servo_three = 135
    return [servo_one, servo_two, servo_three]



def find_center_of_hand(frame):
    frame2 = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    frame2 = cv.GaussianBlur(frame2, (5,5),0)
    frame2 = cv.cvtColor(frame2,cv.COLOR_RGB2HSV)
    lower_blue = np.array([0,15,10])
    upper_blue = np.array([180,255,30])
    mask = cv.inRange(frame2, lower_blue, upper_blue)
    _,contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    center_x = 1
    center_y = 1
    w = 1
    h = 1
    if len(contours) != 0:
        c = max(contours, key = cv.contourArea)
        cv.drawContours(frame, c, -1, (0, 255, 0), 3)
        x,y,w,h = cv.boundingRect(c)
        center_x = (x + (x+w))/2
        center_y = (y + (y+h))/2
        cv.circle(frame, (int(center_x),int(center_y) ), 11, (229,66,244), -1)
    return center_x, center_y, w, h




def save_file(data, gest):
    x_file = open("/home/hussam/Desktop/train_gesture/ges2/" + "xdata5" +".txt", "a")
    x_file.write(data)
    x_file.close()
    y_file = open("/home/hussam/Desktop/train_gesture/ges2/" + "ydata5" +".txt", "a")
    y_file.write(gest)
    y_file.close()
    print("\nSAVED Data\n")
