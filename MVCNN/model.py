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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#tf.logging.set_verbosity(tf.logging.INFO)

def vgg(features, labels, mode):
    # In   put layer, change 56 to whatever the dimensions of the input images are
    input_layer = tf.reshape(features['x'], [-1, 64, 64, 1])

    # Conv Layer #1
    conv1 =tf.compat.v1.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # Conv Layer #2
    conv2 = tf.compat.v1.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

     # Conv Layer #1
    conv3 = tf.compat.v1.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # Conv Layer #2
    conv4 = tf.compat.v1.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    
    # Pooling Layer #2
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

     # Conv Layer #1
    conv5 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # Conv Layer #2
    conv6 = tf.compat.v1.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

    conv7 = tf.compat.v1.layers.conv2d_transpose(inputs=conv6,filters=128, kernel_size=[3,3], padding = 'same', activation =tf.nn.relu)
    
    # Pooling Layer #2
    pool3 = tf.compat.v1.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool3, [-1, 8 * 8 * 128])
    dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.compat.v1.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.compat.v1.layers.dense(inputs=dropout, units=2)

    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {"accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


#images, labels = load(64)
#print(images0)
# import numpy as np
# import tensorflow as tf
import chairs_dataset
# import model as myvgg


#tf.logging.set_verbosity(tf.logging.INFO)

#how many samples will be part of the train slice, the rest will be test data
train_slice = 0.8

#load chairs dataset
images, labels = chairs_dataset.load(64)

l0 = labels[0]
l1 = labels[1]
l2 = labels[2]
l3 = labels[3]
l4 = labels[4]
l5 = labels[5]

i0 = images[0]
i1 = images[1]
i2 = images[2]
i3 = images[3]
i4 = images[4]
i5 = images[5]

#first we calculate the ID that will split the dataset between training samples and test samples
dataset_length = images[0].shape[0]
sliceId = int(dataset_length * train_slice)

train_images = []
train_labels = []

#we get the train images and labels
train_images.append(i0[:sliceId])
train_images.append(i1[:sliceId])
train_images.append(i2[:sliceId])
train_images.append(i3[:sliceId])
train_images.append(i4[:sliceId])
train_images.append(i5[:sliceId])

train_labels.append(l0[:sliceId])
train_labels.append(l1[:sliceId])
train_labels.append(l2[:sliceId])
train_labels.append(l3[:sliceId])
train_labels.append(l4[:sliceId])
train_labels.append(l5[:sliceId])

test_images = []
test_labels = []
#then the rest are test images and labels
test_images.append(i0[sliceId:])
test_images.append(i1[sliceId:])
test_images.append(i2[sliceId:])
test_images.append(i3[sliceId:])
test_images.append(i4[sliceId:])
test_images.append(i5[sliceId:])

test_labels.append(l0[sliceId:])
test_labels.append(l1[sliceId:])
test_labels.append(l2[sliceId:])
test_labels.append(l3[sliceId:])
test_labels.append(l4[sliceId:])
test_labels.append(l5[sliceId:])

for view in [0, 1, 2, 3, 4, 5]:
    id = [0, 1, 2, 3, 4, 5].index(view)

    classifier = tf.estimator.Estimator(
        model_fn=vgg, model_dir="checkpoint/"+str(view)+"/")

    tensors_to_log = {"probabilities": "softmax_tensor"}

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(train_images[id])},
                                                                  y=np.array(train_labels[id]),
                                                                  num_epochs=100,
                                                                  shuffle=True)

    # logging_hook = tf.estimator.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=2)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(test_images[id])},
                                                                 y=np.array(test_labels[id]),
                                                                 num_epochs=1,
                                                                 shuffle=False)
    classifier.train(train_input_fn)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
