#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-10-11 下午6:57
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
 mnist.train : training data
 mnist.test  : test data
 mnist.validation : validation data
 mnist.train.images : training images -- tensor, an n-dimensional array
 mnist.train.labels : corresponding training labels
"""
# None 表示该维度可以任意长度,input any number of MNIST images
# placeholder 是一个占位符，当我们要求TensorFlow运行一个计算时，我们将其输入。
x = tf.placeholder(tf.float32, [None, 784])

# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations
# 对于机器学习应用，通常模型参数是 Variable。
W = tf.Variable(initial_value=tf.zeros([784, 10]), trainable=True)  # 784 x 10
b = tf.Variable(tf.zeros([1, 10]))  # 1 x 10

# build the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# add a new placeholder to input the correct answers，即实际值
y_correct = tf.placeholder(tf.float32, [None, 10])

# 交叉熵
# tf.reduce_mean : 用于计算 tensor 的某一维度的均值
# tf.reduce_sum  : 用于计算 tensor 的某一维度的和, reduction_indices 用于指定某一维度
# -tf.reduce_sum(y_correct * tf.log(y) 计算交叉熵 unstable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_correct))

# ask TensorFlow to minimize cross_entropy
trainning = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# ================ 以上的工作是将 new operations 添加到 computation graph =================
# gives you back a single operation : trainning

# create an operation to initialize the variables we created
init = tf.initialize_all_variables()
# ================ launch the model in a Session ================

"""
InteractiveSession class :
    which makes TensorFlow more flexible about how you structure your code.
    It allows you to interleave operations which build a computation graph
    with ones that run the graph.

"""
session = tf.InteractiveSession()
# session = tf.Session()
session.run(fetches=init)

# trainning the model
for i in xrange(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    # feeding in the batches data to replace the placeholders
    # x, y_correct 占位符 placeholder
    session.run(trainning, feed_dict={x: batch_x, y_correct: batch_y})

# ================ Evaluating Our Model ================
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_correct, 1))  # bool

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # bool -> tf.float32

accuracy = session.run(accuracy, feed_dict={x: mnist.test.images, y_correct: mnist.test.labels})
print accuracy
