#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Save and Restore a model

@author: MarkLiu
@time  : 16-10-22 下午9:04
"""
import tensorflow as tf
import numpy as np

model_path = "model/somemodel.ckpt"

# some model param
W = tf.Variable(np.random.rand(), name="weight")
b = tf.Variable(np.random.rand(), name="bias")

# Initializing the variables
init = tf.initialize_all_variables()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    save_path = saver.save(sess, save_path=model_path)
    print "Model saved in file: %s" % save_path
    print "W:", sess.run(W)
    print "b:", sess.run(b)

print "==========="
# Running a new session
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    # Restore model weights from previously saved model
    load_path = saver.restore(sess, model_path)
    print "Model restored from file: %s" % model_path
    print "W:", sess.run(W)
    print "b:", sess.run(b)
