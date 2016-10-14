#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
convolutional neural network

@author: MarkLiu
@time  : 16-10-13 上午10:32
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name="x_images")
y_correct = tf.placeholder(tf.float32, [None, 10], name="correct_labels")

"""
Weight Initialization
"""
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """
    Since we're using ReLU neurons, it is also good practice to initialize them
    with a slightly positive initial bias to avoid "dead neurons".
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial ,name=name)


"""
Convolution and Pooling
"""
def conv2d(x_, W):
    return tf.nn.conv2d(x_, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_):
    return tf.nn.max_pool(x_, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


"""
First Convolutional Layer: 32 features, 5x5 patch
It will consist of convolution, followed by max pooling.
5 x 5: patch size, local receptive fields;
1    : the number of input channels
32   : the number of output channels, features
"""
W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

# To apply the layer, we first reshape x to a 4d tensor， then convolve x_image with the weight tensor
# 28 x 28 : image size
# 1       : one channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

hidden_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
hidden_pool1 = max_pool_2x2(hidden_conv1)

"""
Second Convolutional Layer: 64 features, 5x5 patch
"""
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

hidden_conv2 = tf.nn.relu(conv2d(hidden_pool1, W_conv2) + b_conv2)

hidden_pool2 = max_pool_2x2(hidden_conv2)

"""
Densely Connected Layer
Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024
neurons to allow processing on the entire image. We reshape the tensor from the pooling
layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
"""
W_full_con = weight_variable([7 * 7 * 64, 1024], "W_full_con")
b_full_con = bias_variable([1024], "b_full_con")

hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])

full_con = tf.nn.relu(tf.matmul(hidden_pool2_flat, W_full_con) + b_full_con)

# Dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
full_con_dropout = tf.nn.dropout(full_con, keep_prob=keep_prob)

"""
Readout Layer: 1024 -> 10
"""
W_full_con_out = weight_variable([1024, 10], "W_full_con_out")
b_full_con_out = bias_variable([10], "b_full_con_out")

# build the model
y = tf.nn.softmax(tf.matmul(full_con_dropout, W_full_con_out) + b_full_con_out)

"""
Train and Evaluate the Model
"""
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_correct, name="cross_entropy")

# ask TensorFlow to minimize cross_entropy
# trainning = tf.train.AdagradOptimizer(learning_rate=0.0001).minimize(cross_entropy)         # test accuracy 0.8562
trainning = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)   # test accuracy 0.9808
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_correct, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

init = tf.initialize_all_variables()
session = tf.InteractiveSession()

session.run(init)

# trainning
for i in xrange(40000):
    batch_x, batch_y = mnist.train.next_batch(50)
    if i % 50 == 0:
        print "step %d, training accuracy %g" % \
              (i, accuracy.eval(feed_dict={x: batch_x, y_correct: batch_y, keep_prob: 1}))

    session.run(trainning, feed_dict={x: batch_x, y_correct: batch_y, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_correct: mnist.test.labels, keep_prob: 1.0}))
