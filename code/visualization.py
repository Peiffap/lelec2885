#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 19:23:54 2017

@author: ignacio

Script for testing the performance of the CNN previously trained
The name of the trained model is CNN.h5
"""
import keras
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from keras.models import load_model
from scipy.misc import imread
from skimage.io import imread
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# load a model
model = load_model('CNN.h5')
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds 
#to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#Read the image
img = imread("uno_6.png",as_grey=True).astype(float)
#reshape
image = misc.imresize(img, (28, 28))
#Plot
plt.title('Number:')
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
# normalise it in the same manner as we did for the training data
image = image / 255.0
image = image.reshape(1,28,28,1)

class_idx = 1
for modifier in ['guided', None]:
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                               seed_input=image, backprop_modifier=modifier)
    plt.figure()
    plt.title(modifier)
    plt.imshow(grads, cmap='jet')

print("predicted digit: "+str(model.predict_classes(image)[0]))
