#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
tensorflow mechanics and program standard

@author: MarkLiu
@time  : 16-10-14 下午1:07
"""
import os
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set some model parameters used in this app
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string(flag_name='train_dir', default_value='MNIST_data', docstring='Directory to put the training data.')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_integer('batch_size', 100, 'Batch size, Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('hidden1_units', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('max_step', 40000, 'max trainning step')

# some golbal data
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the input tensors.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return images_placeholder, labels_placeholder


def build_model(images, hidden_layer1_units, hidden_layer2_units):
    """
    构建全连接神经网络模型
    :param images: 输入的图片数据 tensor
    :param hidden_layer1_units: hidden layer units
    :param hidden_layer2_units: hidden layer units
    :return: 模型输出 tensor
    """
    # hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden_layer1_units], stddev=1.0 / math.sqrt(IMAGE_PIXELS)),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([hidden_layer1_units]),
                             name='biases')
        hidden_layer1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_layer1_units, hidden_layer2_units], stddev=1.0 / math.sqrt(hidden_layer1_units)),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([hidden_layer2_units]),
                             name='biases')
        hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_layer2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(hidden_layer2_units)),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        # final output
        logits = tf.matmul(hidden_layer2, weights) + biases

    return logits


def evaluate_loss(logits, correct_labels):
    """
    计算 loss
    :param logits:
    :param correct_labels:
    :return:
    """
    correct_labels = tf.to_int64(correct_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, correct_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def get_feed_dict(data_sets, images_placeholder, labels_placeholder):
    batch_images, batch_labels = data_sets.next_batch(FLAGS.batch_size,
                                                            FLAGS.fake_data)
    feed_dict = {
        images_placeholder: batch_images,
        labels_placeholder: batch_labels
    }
    return feed_dict


def evaluate(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = get_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count * 1.0 / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

def trainning(learning_rate):
    """
    训练模型
    """
    mnist_data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    # use the default graph
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = build_model(images_placeholder,
                                  FLAGS.hidden1_units,
                                  FLAGS.hidden2_units)
        loss = evaluate_loss(logits, labels_placeholder)

        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)

        # 梯度下降最小化 loss
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 评估预测正确的 op
        eval_correct_op = evaluate(predict_out, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        ##### And then after everything is built:

        sess.run(init)

        # start running loop
        for step_i in xrange(FLAGS.max_step):
            feed_dict = get_feed_dict(mnist_data_sets.train, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            # Write the summaries and print an overview fairly often.
            if step_i % 100 == 0:
                print('Step %d: loss = %.2f' % (step_i, loss_value))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step_i)
                summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
            if (step_i + 1) % 1000 == 0 or (step_i + 1) == FLAGS.max_step:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step_i)

                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct_op,
                        images_placeholder,
                        labels_placeholder,
                        mnist_data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct_op,
                        images_placeholder,
                        labels_placeholder,
                        mnist_data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct_op,
                        images_placeholder,
                        labels_placeholder,
                        mnist_data_sets.test)


if __name__ == '__main__':
    trainning(FLAGS.learning_rate)
