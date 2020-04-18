import cv2
import os
import numpy as np
import tensorflow as tf

'''
load chair dataset. Dimension refers to the target dimension of the output image, used to save up memory.
The images are originally 224 x 224.

There are opportunities to improve the dataset by performing image operations to augment the dataset and generating
more negative samples based on the given meshes.
'''
VIEWS = 6  # Total views

# loads the dataset from the folder and creates labels
def load(dimension):
    # list for each view
    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []
    images5 = []

    isPositive = False

    labels0 = []
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    labels5 = []

    ls = 0

    for id, folder in enumerate(["./all-chairs/models/good/",
                                 "./all-chairs/models/bad/"]):
        isPositive = not isPositive

        length = (len(os.listdir(folder)) // VIEWS ) 
        ls += length

        files = os.listdir((folder))
        files = sorted(files)

        for filename in files:

            view = int(filename.split("_")[1].split('.')[0])
            view = view % VIEWS

            # because the images have 1 channel.
            img = cv2.imread(folder+filename, cv2.IMREAD_GRAYSCALE)

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

        if isPositive:
            labels0 = np.ones((length), dtype=np.int)
            labels1 = np.ones((length), dtype=np.int)
            labels2 = np.ones((length), dtype=np.int)
            labels3 = np.ones((length), dtype=np.int)
            labels4 = np.ones((length), dtype=np.int)
            labels5 = np.ones((length), dtype=np.int)
        else:
            labels0 = np.append(labels0, np.zeros((length), dtype=np.int), axis=0 )
            labels1 = np.append(labels1, np.zeros((length), dtype=np.int), axis=0 )
            labels2 = np.append(labels2, np.zeros((length), dtype=np.int), axis=0 )
            labels3 = np.append(labels3, np.zeros((length), dtype=np.int), axis=0 )
            labels4 = np.append(labels4 , np.zeros((length), dtype=np.int), axis=0 )
            labels5 = np.append(labels5, np.zeros((length), dtype=np.int), axis=0 )

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

    #shuffle all
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(images0)
    np.random.seed(seed)
    np.random.shuffle(images1)
    np.random.seed(seed)
    np.random.shuffle(images2)
    np.random.seed(seed)
    np.random.shuffle(images3)
    np.random.seed(seed)
    np.random.shuffle(images4)
    np.random.seed(seed)
    np.random.shuffle(images5)

    np.random.seed(seed)
    np.random.shuffle(labels0)
    np.random.seed(seed)
    np.random.shuffle(labels1)
    np.random.seed(seed)
    np.random.shuffle(labels2)
    np.random.seed(seed)
    np.random.shuffle(labels3)
    np.random.seed(seed)
    np.random.shuffle(labels4)
    np.random.seed(seed)
    np.random.shuffle(labels5)

    # making a list so it's easier to read
    images = [images0, images1, images2, images3, images4, images5]
    labels = [labels0, labels1, labels2, labels3, labels4, labels5]

    return images, labels