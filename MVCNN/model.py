# Base code by the github user kamalkraj
# available at https://github.com/kamalkraj/Tensorflow-Paper-Implementation
# this is an implementation of Lenet http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
# The basic idea to combine three classifiers, one for each view, was extracted from Zhu et al. https://www.sfu.ca/~cza68/papers/zhu_sig17_scsr.pdf
import chairs_dataset
import numpy as np
import tensorflow as tf
import chairs_dataset
import os 
import sys 

def vgg(features, labels, mode):
    # In   put layer, change 56 to whatever the dimensions of the input images are
    input_layer = tf.reshape(features['x'], [-1, 64, 64, 1])

    # Conv Layer #1
    conv1 = tf.compat.v1.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)
    # Conv Layer #2
    conv2 = tf.compat.v1.layers.conv2d(inputs=conv1, filters=16, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Conv Layer #1
    conv3 = tf.compat.v1.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)
    # Conv Layer #2
    conv4 = tf.compat.v1.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Conv Layer #1
    conv5 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)
    # Conv Layer #2
    conv6 = tf.compat.v1.layers.conv2d(inputs=conv5, filters=64, kernel_size=[3, 3], padding='same',
                                       activation=tf.nn.relu)

    conv7 = tf.compat.v1.layers.conv2d_transpose(inputs=conv6, filters=32, kernel_size=[3, 3], padding='same',
                                                 activation=tf.nn.relu)
    # Dense Layer
    pool2_flat = tf.reshape(conv7, [-1, 16 * 16 * 32])
    dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.compat.v1.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.compat.v1.layers.dense(inputs=dropout, units=2)

    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {"accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


