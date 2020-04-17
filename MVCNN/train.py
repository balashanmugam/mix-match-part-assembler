import tensorflow as tf
import keras
import os
from keras.optimizers import Adam
import numpy as np

import chairs_dataset
import model

# split dataset
images, labels = chairs_dataset.load(64)
i0 = images[0]
i1 = images[1]
i2 = images[2]
i3 = images[3]
i4 = images[4]
i5 = images[5]

train_slice = 0.8  # 80 % training set

dataset_length = images[0].shape[0]
sliceId = int(dataset_length * train_slice)

train0 = i0[:sliceId]
train1 = i1[:sliceId]
train2 = i2[:sliceId]
train3 = i3[:sliceId]
train4 = i4[:sliceId]
train5 = i5[:sliceId]

train_label = labels[:sliceId]

test0 = i0[sliceId:]
test1 = i1[sliceId:]
test2 = i2[sliceId:]
test3 = i3[sliceId:]
test4 = i4[sliceId:]
test5 = i5[sliceId:]

# print(len(test5))

test_label = labels[sliceId:]

# print(sliceId)
# print(dataset_length -sliceId)

# model
model = model.mvcnn()

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(lr=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train(image, label):  # image contains 6 views in a list
    with tf.GradientTape() as tape:
        preds = model.call(image)
        loss = loss_object(label, preds)
    #(preds)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    train_loss(loss)
    train_accuracy(label, preds)


@tf.function
def test(image, label):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  preds = model.call(image)
  t_loss = loss_object(label, preds)

  test_loss(t_loss)
  test_accuracy(label, preds)

EPOCH = 5 
w = 0
for e in range(EPOCH):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for i in range(0, sliceId-1):
        train_s = [train0[i], train1[i], train2[i],
                   train3[i], train4[i], train5[i]]
        train_data = tf.convert_to_tensor(train_s, dtype=tf.float32)
        label_data = tf.convert_to_tensor(train_label[i],dtype=tf.float32)
        print(label_data)
        label_data = tf.reshape(label_data,(1,2))

        train(train_data,label_data)

    for i in range(0,dataset_length-sliceId):
        test_s = [test0[i], test1[i], test2[i],
                   test3[i], test4[i], test5[i]]

        test_data = tf.convert_to_tensor(test_s, dtype=tf.float32)
        label_data = tf.convert_to_tensor(test_label[i],dtype=tf.float32)
        label_data = tf.reshape(label_data,(1,2))

        test(test_data,label_data)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    if (e+1) % 3 == 0:
        #save weights
        cpt = "./Checkpoints/model_weights_cp_" + str(w)+ ".h5"
        w = w + 1
        weights = model.save_weights(cpt)
        print("Saving weight:",w)
        
    print(template.format(e + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

    # need to save checkpoints after few epochs