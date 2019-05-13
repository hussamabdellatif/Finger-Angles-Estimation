from random import shuffle
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt

x_train = []
y_train = []

with open("/home/hussam/Desktop/train_gesture/ges1/features.txt") as f:
    for line in f:
        str = line
        x_train.append([float(i) for i in str.split(',')])

with open("/home/hussam/Desktop/train_gesture/ges1/labels.txt") as f:
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

# print(xtrain_new[0])
# print(ytrain_new[0])
# print(x_test[0])
# print(y_test[0])

learning_rate = 0.005
training_dropout = 1
display_step = 1
training_epochs = 1400
batch_size = 350
accuracy_history = []
cost_history = []
valid_accuracy_history = []
valid_cost_history = []
n_samples = len(xtrain_new)
input_nodes = xtrain_newnp.shape[1]
num_labels = 5

h1 = 9
h2 = 5
h3 = 3
output_nodes = num_labels

pkeep = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, input_nodes], name="myInput")

W1 = tf.Variable(tf.truncated_normal([input_nodes, h1],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1]))
y1 = tf.nn.relu((tf.matmul(x,W1)+b1))

W2 = tf.Variable(tf.truncated_normal([h1, h2],stddev=0.1))
b2 = tf.Variable(tf.zeros([h2]))
y2 = tf.nn.relu((tf.matmul(y1,W2)+b2))

W3 = tf.Variable(tf.truncated_normal([h2, h3],stddev=0.1))
b3 = tf.Variable(tf.zeros([h3]))
y3 = tf.nn.relu((tf.matmul(y2,W3)+b3))
y3 = tf.nn.dropout(y3, pkeep)

W4 = tf.Variable(tf.random_normal([h3, num_labels], stddev=0.1))
b4 = tf.Variable(tf.zeros([num_labels]))
y4 = tf.nn.softmax((tf.matmul(y3, W4) + b4), name="myOutput")

y = y4
y_ = tf.placeholder(tf.float32, [None, num_labels])

cost = -tf.reduce_sum(y_ * tf.log(y))
#     #optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_predicition = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples/batch_size)):
            batch_x = input_X[batch * batch_size : (1 + batch) * batch_size]
            batch_y = input_Y[batch * batch_size : (1 + batch) * batch_size]

            sess.run([optimizer], feed_dict={x: batch_x,
                                             y_: batch_y,
                                             pkeep: training_dropout})

        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost],
                                               feed_dict={x: input_X, y_: input_Y,
                                                          pkeep: training_dropout})
            if(abs(train_accuracy - 0.83) < 0.01):
                print("\nEnding Training\n")
                save_path = saver.save(sess, "/home/hussam/Desktop/train_gesture/tmp/model")
                break

            valid_accuracy, valid_newCost = sess.run([accuracy, cost],
                                                     feed_dict={x: input_X_valid,
                                                                y_: input_Y_valid, pkeep: 1})

            print ("Epoch:", epoch, "Acc =", "{:.5f}".format(train_accuracy),
                   "Cost =", "{:.5f}".format(newCost),
                   "Valid_Acc =", "{:.5f}".format(valid_accuracy),
                   "Valid_Cost = ", "{:.5f}".format(valid_newCost))


            # Record the results of the model
            accuracy_history.append(train_accuracy)
            cost_history.append(newCost)
            valid_accuracy_history.append(valid_accuracy)
            valid_cost_history.append(valid_newCost)

            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_history) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0

    print("Optimization Finished!")

    # Plot the accuracy and cost summaries
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

    ax1.plot(accuracy_history, color='b') # blue
    ax1.plot(valid_accuracy_history, color='g') # green
    ax1.set_title('Accuracy')

    ax2.plot(cost_history, color='b')
    ax2.plot(valid_cost_history, color='g')
    ax2.set_title('Cost')

    plt.xlabel('Epochs (x10)')
    plt.show()

# train_step = tf.train.GradientDesabscentOptimizer(learning_rate).minimize(cross_entropy)

# print(len(y_train))
# training_epochs = 100
# training_dropout = 0.9
# display_step = 1
# batch_size = 300
# accuracy_history = []
# cost_history = []
# valid_accuracy
# n_neurons_in_h1 = 15
# n_neurons_in_h2 = 10
# n_neurons_in_h3 = 10
# n_neurons_in_output = 5
# learning_rate = 0.001
#
# n_features = 15
# n_classes = 5
# #
# X = tf.placeholder(tf.float32, [len(x_train), n_features], name='features')
# Y = tf.placeholder(tf.float32, [len(y_train), n_classes], name='labels')
# keep_prob=tf.placeholder(tf.float32,name='drop_prob')
# W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
# b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
#
# y1 = tf.nn.tanh((tf.matmul(X,W1)+b1),name='activationLayer1')
#
# #network parameters(weights and biases) are set and initialized(Layer2)
# W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='weights2')
# b2 = tf.Variable(tf.random_normal([n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='biases2')
# #activation function(sigmoid)
# y2 = tf.nn.sigmoid((tf.matmul(y1,W2)+b2),name='activationLayer2')
#
# #network parameters(weights and biases) are set and initialized(Layer2)
# W3 = tf.Variable(tf.random_normal([n_neurons_in_h2, n_neurons_in_h3],mean=0,stddev=1/np.sqrt(n_features)),name='weights3')
# b3 = tf.Variable(tf.random_normal([n_neurons_in_h3],mean=0,stddev=1/np.sqrt(n_features)),name='biases3')
# #activation function(sigmoid)
# y3 = tf.nn.sigmoid((tf.matmul(y2,W3)+b3),name='activationLayer3')
#
# #output layer weights and biasies
# Wo = tf.Variable(tf.random_normal([n_neurons_in_h3, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='weightsOut')
# bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='biasesOut')
# #activation function(softmax)
# a = tf.nn.softmax((tf.matmul(y3, Wo) + bo), name='activationOutputLayer')
#
#     #cost function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a),reduction_indices=[1]))
#     #optimizer
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#
#
# #compare predicted value from network with the expected value/target
# correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
# #accuracy determination
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
#
# # initialization of all variables
# initial = tf.global_variables_initializer()
#
# #creating a session
# with tf.Session() as sess:
#     sess.run(initial)
#     writer = tf.summary.FileWriter("/home/hussam/Desktop/train_gesture")
#     writer.add_graph(sess.graph)
#     merged_summary = tf.summary.merge_all()
#
#     # training loop over the number of epoches
#     x_batch=x_train
#     y_batch=y_train
#     for epoch in range(training_epochs):
#         # feeding training data/examples
#         sess.run(train_step, feed_dict={X:x_batch , Y:y_batch})
#         # feeding testing data to determine model accuracy
#         y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: x_test})
#         y_true = sess.run(tf.argmax(y_test, 1))
#         summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: x_test, Y: y_test})
#         # write results to summary file
#         writer.add_summary(summary, epoch)
#         # print accuracy for each epoch
#         print('epoch',epoch, acc)
#         print ('---------------')
#         print(y_pred, y_true)
