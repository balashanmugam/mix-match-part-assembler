import numpy as np
import tensorflow as tf
import os
import cv2
from model import vgg


#tf.logging.set_verbosity(tf.logging.INFO)
VIEWS = 6  # Total views

def load_eval(dimension):

    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []
    images5 = []

    ls = 0
    folder = "./evaluate-chairs/" # change it 

    length = len(os.listdir(folder)) // VIEWS
    ls += length

    for filename in os.listdir(folder):

        view = int(filename.split("_")[1].split('.')[0])

        view = view % VIEWS

        img = cv2.imread(folder+filename)


        #This relies on the files being loaded in order. For that to happen, the 0 padding in the file name is crucial.
        #If you do not have that, then you need to change the logic of this loop.
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

def main(*argv):

    #load chairs dataset
    test_images = load_eval(56)


    # test_images = {}
    # #then the rest are test images and labels
    # test_images["Top"] = imagesTop
    # test_images["Front"] = imagesFront
    # test_images["Side"] = imagesSide

    test_evaluations = [[],[],[]]

    for id, view in enumerate([0,1,2,3,4,5]):
        classifier = tf.estimator.Estimator(model_fn=vgg, model_dir="checkpoint/"+str(view)+"/")

        eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(x={"x": np.array(test_images[view])},
                                                           num_epochs=1,
                                                           shuffle=False)

        #The line below returns a generator that has the probability that the tested samples are Positive cases or Negative cases
        eval_results = classifier.predict(input_fn=eval_input_fn)

        #You need to iterate over the generator returned above to display the actual probabilities
        #This line should print something like {'classes': 0, 'probabilities': array([0.91087427, 0.08912573])}
        #the first element of 'probabilities' is the correlation of input with the Negative samples. The second element means positive.
        #If you evaluate multiple samples, just keep iterating over the eval_results generator.
        #eval_instance = next(eval_results)
        #print(eval_instance)

        # This is how you extract the correlation to the positive class of the first element in your evaluation folder
        for eval in eval_results:
            #print("probability that this instance is positive is %3.2f " % eval['probabilities'][1])
            test_evaluations[id].append(eval['probabilities'][1])

    #the probability that the chair is a positive example is given by the minimum of the probabilities from each of the three views
    #in the default configuration sent, the first ten chairs should be negatives (low value) and the ten last chairs should be positives (high value)
    #as can be seen in this quick evaluation, there is roon for inprovement in the algorithm
    evaluation_chairs = np.amin(test_evaluations, axis=0)
    print(evaluation_chairs)


if __name__ == "__main__":
    # Add ops to save and restore all the variables.
    #saver = tf.train.Saver()
    tf.app.run()
