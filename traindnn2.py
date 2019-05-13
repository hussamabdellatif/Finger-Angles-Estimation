from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt

x_train = []
y_train = []


with open("/home/hussam/Desktop/train_gesture/ges2/xdata.txt") as f:
    for line in f:
        str = line
        x_train.append([float(i) for i in str.split(',')])

with open("/home/hussam/Desktop/train_gesture/ges2/ydata.txt") as f:
    for line in f:
        str = line
        y_train.append([int(i) for i in str.split(',')])

ind_list = [i for i in range(len(x_train))]
shuffle(ind_list)
xtrain_new = []
ytrain_new = []

for i in ind_list:
    xtrain_new.append(x_train[i])
    ytrain_new.append(y_train[i])

xtrain_newnp = np.asarray(xtrain_new, dtype=np.float32)
ytrain_newnp = np.asarray(ytrain_new, dtype=np.int)
x_test = xtrain_new[0:150]
y_test = ytrain_new[0:150]
x_testnp = np.asarray(x_test, dtype=np.float32)
y_testnp = np.asarray(y_test, dtype=np.int)


split = int(len(y_testnp)/2)

train_size = xtrain_newnp.shape[0]
n_samples = ytrain_newnp.shape[0]

input_X = xtrain_newnp
input_Y = ytrain_newnp
input_X_valid = x_testnp[:split]
input_Y_valid = y_testnp[:split]
input_X_test = x_testnp[split:]
input_Y_test = y_testnp[split:]


model = Sequential()
model.add(Dense(13,input_dim=19, activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(input_X, input_Y, epochs=100, batch_size=1000)
scores = model.evaluate(input_X, input_Y)
print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
model.save('')
