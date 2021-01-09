import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# dataset used: https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'skin_dataset/train/',
        classes = ['benign', 'malignant'],
        target_size=(200, 200),
        batch_size=329,
        # Use binary labels
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'skin_dataset/test/',
        classes = ['benign', 'malignant'],
        target_size=(200, 200),
        batch_size=82,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(220, 220, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
#                                     tf.keras.layers.Dense(128, activation=tf.nn.relu), 
#                                     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
# 
model.compile(loss='binary_crossentropy',
optimizer='Adam', metrics='accuracy')

history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=10, verbose=1, validation_data=validation_generator, validation_steps=8)

model.evaluate(validation_generator)

model.save('SkinCD')