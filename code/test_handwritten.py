#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:17:10 2017

@author: ignacio

Script used to test the CNN trained with handwritten numbers

"""

import keras
from keras.datasets import mnist
from keras.models import load_model
from scipy.misc import imread
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

# load a model
model = load_model('firstCNN.h5')

# load the handwritten number
img = imread("dos.png",as_grey=True).astype(float)


#reshape
image = misc.imresize(img, (28, 28))
plt.title('Number:')
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
# normalise it in the same manner as we did for the training data
image = image / 255.0
image = image.reshape(1,28,28,1)




# forward propagate and print index of most likely class 
# (for MNIST this corresponds one-to-one with the digit)
print("predicted digit: "+str(model.predict_classes(image)[0]))
