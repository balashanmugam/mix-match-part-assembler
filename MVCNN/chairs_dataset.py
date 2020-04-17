import cv2
import os
import numpy as np

'''
load chair dataset. Dimension refers to the target dimension of the output image, used to save up memory.
The images are originally 224 x 224.

There are opportunities to improve the dataset by performing image operations to augment the dataset and generating
more negative samples based on the given meshes.
'''
VIEWS = 6  # Total views


def load(dimension):
    # list for each view
    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []
    images5 = []

    isPositive = False

    labels = [[]]

    ls = 0
    
    for id, folder in enumerate(["./Data/data/good1/", "./Data/data/bad1/"]):
        isPositive = not isPositive

        length = len(os.listdir(folder)) // VIEWS
        ls += length

        files = os.listdir((folder))
        files = sorted(files)
        #print(files)
    
        for filename in files:

            view = int(filename.split("_")[1].split('.')[0])
            view = view % VIEWS

            img = cv2.imread(folder+filename,cv2.IMREAD_GRAYSCALE)
 
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
                    #print("Label added:", filename)
                    a = [1., 0.]
                    labels.append(a)
                else:
                    a = [0., 1.]
                    labels.append(a)
                    #print("Negative Label added:", filename)

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
    np.random.shuffle(labels)

    images = [images0, images1, images2, images3, images4, images5]
    
    #print(labels)
    return images, labels


# def runtime_load_test():
#     import time
#     start_time = time.time()
#     images, labels = load(56)
#     print("--- %s min ---" % ((time.time() - start_time) / 60))
#     #print(imagesTop.shape[0])


#images, labels = load(64)
#print(images0)
