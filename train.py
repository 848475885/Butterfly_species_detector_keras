"""Code to train a network to classify between different species of butterflies using Keras(Tensorflow) and Python 
Author:Bhavan Vasu,
Graduate research assistant,
Real time and computer vision lab,
Rochester Institute of technology,New york-14623
email:bxv7657@rit.edu"""
from keras.preprocessing.image import ImageDataGenerator
from tflearn.data_utils import image_preloader
from keras.applications.vgg16 import VGG16
from keras.layers.core import  Dense
from keras import backend as K
import matplotlib.pylab as plt
from keras.models import Model
from PIL import Image    
import numpy as np
import h5py


K.set_learning_phase(1)
batch_size=32
train_data_dir = './data/'
nb_train_samples = 782
nb_validation_samples = 50
epochs = 3




#Using a VGG16 pre-trained on ImageNet
resmo = VGG16(weights='imagenet', include_top=True)

#Freezing all the above layers
for layer in (resmo.layers):
  layer.trainable = False

#Adding a new Dense layer with 10 output nodes for 10 species of butterflies
x = Dense(10, activation='softmax', name='predictions')(resmo.layers[-2].output)

#Redeffining a new model with 10 output class nodes
my_model = Model(inputs=resmo.input,outputs=(x))
my_model.summary()


my_model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=['accuracy'])
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Data augumentation for increasing the number of training samples
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#Data directory containing all the training images in subfolders 
train_generator = train_datagen.flow_from_directory(
    '/home/bxv7657/but/data/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    '/home/bxv7657/but/validation/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

#Training the network with 782 test and 50 validation images from http://www.josiahwang.com/dataset/leedsbutterfly/

my_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle=True)

# Save model for testing
my_model.save_weights('butterflyvggs_weights.h5')

