# Base code by the github user kamalkraj
# available at https://github.com/kamalkraj/Tensorflow-Paper-Implementation
# this is an implementation of Lenet http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
# The basic idea to combine three classifiers, one for each view, was extracted from Zhu et al. https://www.sfu.ca/~cza68/papers/zhu_sig17_scsr.pdf
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, AveragePooling2D, Dropout, GlobalAveragePooling2D, Softmax
import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
import chairs_dataset
import tensorflow as tf
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.logging.set_verbosity(tf.logging.INFO)


class vgg(Model):
    def __init__(self):
        super(vgg, self).__init__()
        self.mp = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c1 = Conv2D(filters=64, kernel_size=(
            3, 3), padding='same', activation='relu')
        self.c2 = Conv2D(filters=64, kernel_size=(
            3, 3), padding='same', activation='relu')

        self.c3 = Conv2D(filters=128, kernel_size=(
            3, 3), padding='same', activation='relu')
        self.c4 = Conv2D(filters=128, kernel_size=(
            3, 3), padding='same', activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(
            3, 3), padding='same', activation='relu')
        self.c6 = Conv2D(filters=256, kernel_size=(
            3, 3), padding='same', activation='relu')
        self.c7 = Conv2D(filters=256, kernel_size=(
            3, 3), padding='same', activation='relu')

        self.flat = Flatten()

    def call(self, x):
        x = tf.reshape(x, [-1, 64, 64, 1])
        x = self.c1(x)
        x = self.c2(x)
        x = self.mp(x)
        #print(x.shape)

        x = self.c3(x)
        x = self.c4(x)
        x = self.mp(x)
        #print(x.shape)

        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.mp(x)
        #print(x.shape)

        x = self.flat(x)
        #print(x.shape)

        return x


class mvcnn(Model):
    def __init__(self):
        super(mvcnn, self).__init__()
        self.vgg0 = vgg()
        self.vgg1 = vgg()
        self.vgg2 = vgg()
        self.vgg3 = vgg()
        self.vgg4 = vgg()
        self.vgg5 = vgg()

        # dense pooling
        # average pooling

        self.ap = GlobalAveragePooling2D()
        self.d1 = Dense(1536, activation='relu')
        self.d2 = Dense(1536, activation='relu')
        self.final = Dense(2)  # two classes classification

        self.drop1 = Dropout(0.5)
        self.drop2 = Dropout(0.3)
        self.soft = Softmax()

    def call(self, x):
        #print(x[0].shape)
        x0 = self.vgg0.call(x[0])
        x1 = self.vgg1.call(x[1])
        x2 = self.vgg2.call(x[2])
        x3 = self.vgg3.call(x[3])
        x4 = self.vgg4.call(x[4])
        x5 = self.vgg5.call(x[5])

        all_feats = tf.stack([x0, x1, x2, x3, x4, x5])

        #print('Before:', all_feats.shape)
        all_feats = tf.reshape(all_feats, (-1, 8, 8, 256))
        #print('After:',all_feats.shape)
        out = self.ap(all_feats)
        #print(out.shape)
        out = tf.reshape(out, (-1, 1536))
        out = self.drop1(out)
        out = self.d1(out)
        out = self.drop2(out)
        out = self.d2(out)

        out = self.final(out)
        out = self.soft(out)
        #print('Final: ',out[0])
        return out



