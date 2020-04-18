import numpy as np
import tensorflow as tf
import os
import cv2
from model import vgg

VIEWS = 6  # Total views

# loads the evaluation images
def load_eval(dimension):
    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []
    images5 = []

    ls = 200

    # change  before running
    folder = "./MVCNN/Data/images/c/"

    length = len(os.listdir(folder)) // VIEWS
    files = os.listdir((folder))
    files = sorted(files)

    for filename in files:

        view = int(filename.split("_")[1].split('.')[0])

        view = view % VIEWS

        img = cv2.imread(folder + filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if view == 0:
                images0.append(img / 255.)
            elif view == 1:
                images1.append(img / 255.)
            elif view == 2:
                images2.append(img / 255.)
            elif view == 3:
                images3.append(img / 255.)
            elif view == 4:
                images4.append(img / 255.)
            else:
                images5.append(img / 255.)

    images0 = np.array(images0)
    images1 = np.array(images1)
    images2 = np.array(images2)
    images3 = np.array(images3)
    images4 = np.array(images4)
    images5 = np.array(images5)

    images0 = np.reshape(images0, (ls, dimension * dimension))
    images1 = np.reshape(images1, (ls, dimension * dimension))
    images2 = np.reshape(images2, (ls, dimension * dimension))
    images3 = np.reshape(images3, (ls, dimension * dimension))
    images4 = np.reshape(images4, (ls, dimension * dimension))
    images5 = np.reshape(images5, (ls, dimension * dimension))

    images = [images0, images1, images2, images3, images4, images5]


    return images


#load chairs dataset
test_images = load_eval(64)

test_evaluations = [[], [], [], [], [], []]

for id, view in enumerate([0,1,2,3,4,5]):
    classifier = tf.estimator.Estimator(model_fn=vgg, model_dir="checkpoint/"+str(view)+"/")

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(test_images[id])},
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = classifier.predict(input_fn=eval_input_fn)

    for eval in eval_results:
        #print("probability that this instance is positive is %3.2f " % eval['probabilities'][1])
        test_evaluations[id].append(eval['probabilities'][1])

evaluation_chairs = np.amin(test_evaluations, axis=0)

# print results
print(len(evaluation_chairs))
print("______")
print (np.where(evaluation_chairs > 0.99))
print(np.sort(evaluation_chairs))