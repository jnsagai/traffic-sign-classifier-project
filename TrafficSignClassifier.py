# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:04:41 2020

@author: jnnascimento
"""


# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = train['features'].shape[0]
n_validation = valid['features'].shape[0]
n_test = test['features'].shape[0]
image_shape = train['features'].shape[1:3]
n_classes = len(np.unique(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# Convert all the images to YUV
# X_train_pre = []
# for i in range(n_train):
#     X_train_pre.append(cv2.cvtColor(X_train[i], cv2.COLOR_RGB2YUV))

# X_valid_pre = []
# for i in range(n_validation):
#     X_valid_pre.append(cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2YUV))

# X_test_pre = []
# for i in range(n_test):
#     X_test_pre.append(cv2.cvtColor(X_test[i], cv2.COLOR_RGB2YUV))

# X_train_pre = np.array([cv2.cvtColor(X_train[i], cv2.COLOR_RGB2YUV) for i in range(n_train)])
# X_valid_pre = np.array([cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2YUV) for i in range(n_validation)])
# X_test_pre = np.array([cv2.cvtColor(X_test[i], cv2.COLOR_RGB2YUV) for i in range(n_test)])

# Lets consider only the Y channel
# X_train_pre = (X_train_pre[:,:,:,:1] - 128) / 128
# X_valid_pre = (X_valid_pre[:,:,:,:1] - 128) / 128
# X_test_pre = (X_test_pre[:,:,:,:1] - 128) / 128

X_train_pre = X_train
X_valid_pre = X_valid
X_test_pre = X_test

# X_train_pre = (X_train - 128) / 128
# X_valid_pre = (X_valid - 128) / 128
# X_test_pre = (X_test - 128) / 128

# Get the most frequent class in training dataset
most_freq_class = np.bincount(y_train).argmax()
max_samples = np.bincount(y_train)[most_freq_class]

# Get the most frequent class in validation dataset
most_freq_class_valid = np.bincount(y_valid).argmax()
max_samples_valid = np.bincount(y_valid)[most_freq_class_valid]

for i in range(n_classes):      
    # Get the difference between the class with most samples and this class    
    diff = max_samples - np.bincount(y_train)[i]
    
    # Get all the samples from the class
    sample_index = np.where(y_train == i)
    sample_class = X_train_pre[sample_index]
    
    # Get "diff" random number of samples from the class
    new_samples = sample_class[np.random.randint(len(sample_class), size = diff)]
    
    # Stack the new_samples in the original train input and labels
    X_train_pre = np.vstack((X_train_pre, new_samples))
    y_train = np.append(y_train, np.repeat(i, diff))
    
for i in range(n_classes):      
    # Get the difference between the class with most samples and this class    
    diff = max_samples - np.bincount(y_valid)[i]
    
    # Get all the samples from the class
    sample_index = np.where(y_valid == i)
    sample_class = X_valid_pre[sample_index]
    
    # Get "diff" random number of samples from the class
    new_samples = sample_class[np.random.randint(len(sample_class), size = diff)]
    
    # Stack the new_samples in the original train input and labels
    X_valid_pre = np.vstack((X_valid_pre, new_samples))
    y_valid = np.append(y_valid, np.repeat(i, diff))
    

#Shuffle the training data
X_train_pre, y_train = shuffle(X_train_pre, y_train)
X_valid_pre, y_valid = shuffle(X_valid_pre, y_valid)

# Plotting label data metrics
# Plot a random training image
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))
ax1.set_title('Random training')
ax1.imshow(X_train_pre[int(np.random.randint(n_train, size=1))], cmap = 'gray')

# Plot histogram of training classes distribution
ax2.set_title('Training classes distribution')
ax2.hist(y_train, bins = n_classes)

# Plot histogram of validation labels distribution
ax3.set_title('Validation classes distribution')
ax3.hist(y_valid, bins = n_classes)

# Plot histogram of testing classes distribution
ax3.set_title('Testing classes distribution')
ax4.hist(test['labels'], bins = n_classes)

### Define your architecture here.
### Feel free to use as many code cells as needed.
"""
Setup TensorFlow
The EPOCH and BATCH_SIZE values affect the training speed and model accuracy.

You do not need to modify this section.
"""
EPOCHS = 10
BATCH_SIZE = 128

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.03
    
    # Filters hyperparameters
    # filter_size_height = 5
    # filter_size_width = 5
    """
    weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
    bias = tf.Variable(tf.zeros(k_output))
    
    Given:
    our input layer has a width of W and a height of H
    our convolutional layer has a filter size F
    we have a stride of S
    a padding of P
    and the number of filters K
    
    W_out = [(W−F+2P)/S] + 1
    H_out = [(H-F+2P)/S] + 1
    """
    weights = {
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 100], mean = mu, stddev = sigma)),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 100, 150], mean = mu, stddev = sigma)),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 150, 250], mean = mu, stddev = sigma)),
        'wd1': tf.Variable(tf.truncated_normal([1000, 300], mean = mu, stddev = sigma)),
        'wd2': tf.Variable(tf.truncated_normal([300, 200], mean = mu, stddev = sigma)),
        'out': tf.Variable(tf.truncated_normal([200, n_classes], mean = mu, stddev = sigma))}
    
    biases = {
        'bc1': tf.Variable(tf.truncated_normal([100], mean = mu, stddev = sigma)),
        'bc2': tf.Variable(tf.truncated_normal([150], mean = mu, stddev = sigma)),
        'bc3': tf.Variable(tf.truncated_normal([250], mean = mu, stddev = sigma)),
        'bd1': tf.Variable(tf.truncated_normal([300], mean = mu, stddev = sigma)),
        'bd2': tf.Variable(tf.truncated_normal([200], mean = mu, stddev = sigma)),
        'out': tf.Variable(tf.truncated_normal([n_classes], mean = mu, stddev = sigma))}
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x100.
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides = [1, 1, 1, 1], padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # TODO: Pooling. Input = 28x28x100. Output = 14x14x100.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Input = 14x14x100. Output = 12x12x150.
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides = [1, 1, 1, 1], padding = 'VALID')
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 12x12x150. Output = 6x6x150.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    
    # TODO: Layer 3: Convolutional. Input = 6x6x150. Output = 4x4x250.
    conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides = [1, 1, 1, 1], padding = 'VALID')
    conv3 = tf.nn.bias_add(conv3, biases['bc3'])
    
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)

    # TODO: Pooling. Input = 4x4x250. Output = 2x2x250.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Flatten. Input = 2x2x250. Output = 400.
    fc1 = flatten(conv3)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 300.
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 300. Output = 200.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 200. Output = 43.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])    
  
    return logits

"""
Features and Labels
Train LeNet to classify MNIST data.

x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.

You do not need to modify this section.
"""
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

"""
Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.

You do not need to modify this section.
"""
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

"""
Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.

You do not need to modify this section.
"""

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data_in, y_data_in):
    num_examples = len(X_data_in)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data_in[offset:offset+BATCH_SIZE], y_data_in[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

"""
Train the Model
Run the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.

You do not need to modify this section.
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_pre)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_pre, y_train = shuffle(X_train_pre, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_pre[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid_pre, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
"""
Evaluate the Model
Once you are completely satisfied with your model, evaluate the performance of the model on the test set.

Be sure to only do this once!

If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.

You do not need to modify this section.
"""
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_pre, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))