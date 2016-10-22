#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
linear regression

@author: MarkLiu
@time  : 16-10-22 下午8:32
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.rand(), name="weight")
b = tf.Variable(np.random.rand(), name="bias")

y_predict = tf.mul(X, W) + b

loss = tf.reduce_mean(tf.pow(y_predict - Y, 2) / (2 * n_samples))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: x, Y: y})

        if epoch % 100 == 0:
            lossValue = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
            print "Epoch: %d, loss = %.4f" % (epoch, lossValue)

    print "Optimization Finished!"
    training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print "Training cost = ", training_cost

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
