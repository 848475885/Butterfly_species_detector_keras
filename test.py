"""Code to test an image on a network trained to classify between different species of butterflies using Keras(Tensorflow) and Python 
Author:Bhavan Vasu,
Graduate research assistant,
Real time and computer vision lab,
Rochester Institute of technology,New york-14623
email:bxv7657@rit.edu"""
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense
import matplotlib.pyplot as plt
from keras.models import Model 
from keras import optimizers
import numpy as np
import random 
import h5py
import ast
import os

#Loading our single test image: This can be easily adapted to work on the video frames from the robot
img_path = "/home/bxv7657/but/0030030.png"
im2 = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
im = np.expand_dims(im2, axis=0)

#Using a VGG16 pre-trained on ImageNet
resmo = VGG16(weights=None, include_top=True)

#Freezing all the above layers
for layer in (resmo.layers):
  layer.trainable = False

#Adding a new Dense layer with 10 output nodes for 10 species of butterflies
x = Dense(10, activation='softmax', name='predictions')(resmo.layers[-2].output)

#Redeffining a new model with 10 output class nodes
my_model = Model(inputs=resmo.input,outputs=(x))
my_model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=['accuracy'])

#Loading weight file generated after running train.py
my_model.load_weights('butterflyvgg_weights.h5')
my_model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=['accuracy'])
preds=my_model.predict(im)

#Printing the name of species from the label.txt file
y_classes = np.argmax(preds)
with open('label.txt') as class_file:
       class_dict = ast.literal_eval(class_file.read())
fig, ax = plt.subplots()
ax.imshow(im2)
x.set_title(class_dict[y_classes])
plt.show()
