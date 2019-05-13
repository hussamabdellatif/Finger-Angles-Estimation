import cv2 as cv
import os

path = '/home/hussam/Desktop/hands/'
new_path = '/home/hussam/Desktop/Project/Data/'
itr = 1

for file in os.listdir(path):
    try:
        img = cv.imread(path+file)
        dim = (900,900)
        resized = cv.resize(img,dim)
        name = str(itr) + '.jpg'
        print(name)
        cv.imwrite(new_path+name,resized)
        itr += 1
    except Exception as e:
        print(str(e))
