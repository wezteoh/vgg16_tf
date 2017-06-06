__author__ = "Wei Zhen Teoh"

# This is a tensorflow model of the 16-layer convolutional neural network used 
# by the VGG team in the ILSVRC-2014 competition
# I wrote up this model with strong reference to Davi Frossard's post on
# https://www.cs.toronto.edu/~frossard/post/vgg16/

import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
import tensorflow as tf

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


# Define network architecture
parameters = []

# input
inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])

# zero-mean input
mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
recentred_inputs = inputs-mean

# conv1_1
W_conv1_1 = weight_variable([3, 3, 3, 64])
b_conv1_1 = bias_variable([64])
conv1_1 = tf.nn.relu(conv2d(recentred_inputs, W_conv1_1) + b_conv1_1)
parameters += [W_conv1_1, b_conv1_1]

# conv1_2
W_conv1_2 = weight_variable([3, 3, 64, 64])
b_conv1_2 = bias_variable([64])
conv1_2 = tf.nn.relu(conv2d(conv1_1, W_conv1_2) + b_conv1_2)
parameters += [W_conv1_2, b_conv1_2]

# pool1
pool1 = max_pool_2x2(conv1_2)

# conv2_1
W_conv2_1 = weight_variable([3, 3, 64, 128])
b_conv2_1 = bias_variable([128])
conv2_1 = tf.nn.relu(conv2d(pool1, W_conv2_1) + b_conv2_1)
parameters += [W_conv2_1, b_conv2_1]

# conv2_2
W_conv2_2 = weight_variable([3, 3, 128, 128])
b_conv2_2 = bias_variable([128])
conv2_2 = tf.nn.relu(conv2d(conv2_1, W_conv2_2) + b_conv2_2)
parameters += [W_conv2_2, b_conv2_2]        

# pool2
pool2 = max_pool_2x2(conv2_2)

# conv3_1
W_conv3_1 = weight_variable([3, 3, 128, 256])
b_conv3_1 = bias_variable([256])
conv3_1 = tf.nn.relu(conv2d(pool2, W_conv3_1) + b_conv3_1)
parameters += [W_conv3_1, b_conv3_1]   

# conv3_2
W_conv3_2 = weight_variable([3, 3, 256, 256])
b_conv3_2 = bias_variable([256])
conv3_2 = tf.nn.relu(conv2d(conv3_1, W_conv3_2) + b_conv3_2)
parameters += [W_conv3_2, b_conv3_2]   

# conv3_3
W_conv3_3 = weight_variable([3, 3, 256, 256])
b_conv3_3 = bias_variable([256])
conv3_3 = tf.nn.relu(conv2d(conv3_2, W_conv3_3) + b_conv3_3)
parameters += [W_conv3_3, b_conv3_3]   

# pool3
pool3 = max_pool_2x2(conv3_3)        

# conv4_1
W_conv4_1 = weight_variable([3, 3, 256, 512])
b_conv4_1 = bias_variable([512])
conv4_1 = tf.nn.relu(conv2d(pool3, W_conv4_1) + b_conv4_1)
parameters += [W_conv4_1, b_conv4_1]        

# conv4_2
W_conv4_2 = weight_variable([3, 3, 512, 512])
b_conv4_2 = bias_variable([512])
conv4_2 = tf.nn.relu(conv2d(conv4_1, W_conv4_2) + b_conv4_2)
parameters += [W_conv4_2, b_conv4_2]   

# conv4_3
W_conv4_3 = weight_variable([3, 3, 512, 512])
b_conv4_3 = bias_variable([512])
conv4_3 = tf.nn.relu(conv2d(conv4_2, W_conv4_3) + b_conv4_3)
parameters += [W_conv4_3, b_conv4_3]

# pool4
pool4 = max_pool_2x2(conv4_3) 

# conv5_1
W_conv5_1 = weight_variable([3, 3, 512, 512])
b_conv5_1 = bias_variable([512])
conv5_1 = tf.nn.relu(conv2d(pool4, W_conv5_1) + b_conv5_1)
parameters += [W_conv5_1, b_conv5_1]

# conv5_2
W_conv5_2 = weight_variable([3, 3, 512, 512])
b_conv5_2 = bias_variable([512])
conv5_2 = tf.nn.relu(conv2d(conv5_1, W_conv5_2) + b_conv5_2)
parameters += [W_conv5_2, b_conv5_2]

# conv5_3
W_conv5_3 = weight_variable([3, 3, 512, 512])
b_conv5_3 = bias_variable([512])
conv5_3 = tf.nn.relu(conv2d(conv5_2, W_conv5_3) + b_conv5_3)
parameters += [W_conv5_3, b_conv5_3]

# pool5
pool5 = max_pool_2x2(conv5_3) 

# fc1
W_fc1 = weight_variable([7 * 7 * 512, 4096])
b_fc1 = bias_variable([4096])
pool5_flat = tf.reshape(pool5, [-1, 7*7*512])
fc1 = tf.nn.relu(tf.matmul(pool5_flat, W_fc1) + b_fc1)
parameters += [W_fc1, b_fc1]

# fc2
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
parameters += [W_fc2, b_fc2]

# fc3
W_fc3 = weight_variable([4096, 1000])
b_fc3 = bias_variable([1000])
fc3 = tf.matmul(fc2, W_fc3) + b_fc3
parameters += [W_fc3, b_fc3]

# softmax
probs = tf.nn.softmax(fc3)


# Test run
"""
from imagenet_classes import class_names

if __name__ == '__main__':
    with tf.Session() as sess:
        # load weights
        weights = np.load('vgg16_weights.npz')
        keys = sorted(weights.keys())
        for i,k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(parameters[i].assign(weights[k]))
        
        # test image
        img1 = imread('laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(probs, feed_dict={inputs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print (class_names[p], prob[p])
"""









