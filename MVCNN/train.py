import numpy as np
import tensorflow as tf
import chairs_dataset
from model import vgg

#how many samples will be part of the train slice, the rest will be test data
train_slice = 0.75

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

# train the model
for view in [0, 1, 2, 3, 4, 5]:
    id = [0, 1, 2, 3, 4, 5].index(view)

    classifier = tf.estimator.Estimator(
        model_fn=vgg, model_dir="checkpoint/"+str(view)+"/")

    tensors_to_log = {"probabilities": "softmax_tensor"}

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(train_images[id])},
                                                                  y=np.array(train_labels[id]),
                                                                  num_epochs=20,
                                                                  shuffle=True)
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(test_images[id])},
                                                                 y=np.array(test_labels[id]),
                                                                 num_epochs=1,
                                                                 shuffle=False)
    classifier.train(input_fn=train_input_fn, steps=10000)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
