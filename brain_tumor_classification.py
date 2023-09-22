# -*- coding: utf-8 -*-
"""brain_tumor_classification
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import load_img

# Read Dataset
train_image = "/Users/selvi/Documents\workspace\Brain tumor classification\Brain Tumor\Training/"
test_image = "/Users/selvi/Documents\workspace\Brain tumor classification\Brain Tumor\Testing/"

train_paths = []
train_labels = []

for label in os.listdir(train_image):
    for image in os.listdir(os.path.join(train_image, label)):
        train_paths.append(os.path.join(train_image, label, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# Plot class distribution in the training data
plt.figure(figsize=(14, 6))
colors = ['#FF5733', '#FFC300', '#33FF57', '#339CFF']
plt.rcParams.update({'font.size': 14})
plt.pie([len([x for x in train_labels if x == 'pituitary']),
         len([x for x in train_labels if x == 'notumor']),
         len([x for x in train_labels if x == 'meningioma']),
         len([x for x in train_labels if x == 'glioma'])],
        labels=['pituitary', 'notumor', 'meningioma', 'glioma'],
        colors=colors, autopct='%.1f%%', explode=(0.025, 0.025, 0.025, 0.025),
        startangle=30)
plt.title('Class Distribution in Training Data')
plt.show()

test_paths = []
test_labels = []

for label in os.listdir(test_image):
    for image in os.listdir(os.path.join(test_image, label)):
        test_paths.append(os.path.join(test_image, label, image))
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)

# Plot class distribution in the testing data
plt.figure(figsize=(14, 6))
colors = ['#FF5733', '#FFC300', '#33FF57', '#339CFF']
plt.rcParams.update({'font.size': 14})
plt.pie([len([x for x in test_labels if x == 'pituitary']),
         len([x for x in test_labels if x == 'notumor']),
         len([x for x in test_labels if x == 'meningioma']),
         len([x for x in test_labels if x == 'glioma'])],
        labels=['pituitary', 'notumor', 'meningioma', 'glioma'],
        colors=colors, autopct='%.1f%%', explode=(0.025, 0.025, 0.025, 0.025),
        startangle=30)
plt.title('Class Distribution in Testing Data')
plt.show()

# Define data augmentation function using Albumentations library
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    Blur,
)

def augment_image(image):
    augmentation = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=30, p=0.5),
        Blur(blur_limit=(3, 7), p=0.5),
    ])

    augmented_image = augmentation(image=np.array(image))['image']
    augmented_image = augmented_image / 255.0
    return augmented_image

# Display a sample of augmented images
IMAGE_SIZE = 256

def open_and_augment_images(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

sample_paths = train_paths[50:59]
sample_labels = train_labels[50:59]

sample_images = open_and_augment_images(sample_paths)

fig = plt.figure(figsize=(12, 6))
for x in range(1, 9):
    fig.add_subplot(2, 4, x)
    plt.axis('off')
    plt.title(sample_labels[x])
    plt.imshow(sample_images[x])
plt.title('Sample Augmented Images')
plt.show()

# Data Generator
unique_labels = os.listdir(train_image)

def encode_label(labels):
    encoded = [unique_labels.index(x) for x in labels]
    return np.array(encoded)

def decode_label(labels):
    decoded = [unique_labels[x] for x in labels]
    return np.array(decoded)

def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_and_augment_images(batch_paths)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_label(batch_labels)
            yield batch_images, batch_labels

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the ResNet-50 base model
base_model = ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Set all layers to non-trainable
for layer in base_model.layers:
    layer.trainable = False

# Create a sequential model
model = Sequential()

# Add the ResNet-50 base model
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(unique_labels), activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

#Train Model

batch_size = 32
steps = int(len(train_paths)/batch_size)
epochs = 100
history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps)

plt.figure(figsize=(8,4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
plt.plot(history.history['loss'], '.r-', linewidth=2)
plt.title('Model Training History')
plt.xlabel('epoch')
plt.xticks([x for x in range(epochs)])
plt.legend(['Accuracy', 'Loss'], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

#Evaluate Model with Test Samples

batch_size = 32
steps = int(len(test_paths)/batch_size)
y_pred = []
y_true = []
for x,y in tqdm(datagen(test_paths, test_labels, batch_size=batch_size, epochs=1), total=steps):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    for i in decode_label(pred):
        y_pred.append(i)
    for i in decode_label(y):
        y_true.append(i)

print(classification_report(y_true, y_pred))
