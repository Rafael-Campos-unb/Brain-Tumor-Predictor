import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_img = []
train_labels = []

test_img = []
test_labels = []

path_train = 'C:/BrainTumorPredictor/Training/'
path_test = 'C:/BrainTumorPredictor/Testing/'
img_size = 300

for i in os.listdir(path_train):
    for j in os.listdir(path_train + i):
        train_img.append(cv2.resize(cv2.imread(path_train + i + '/' + j), (img_size, img_size)))
        train_labels.append(i)

for i in os.listdir(path_test):
    for j in os.listdir(path_test + i):
        test_img.append(cv2.resize(cv2.imread(path_test + i + '/' + j), (img_size, img_size)))
        test_labels.append(i)

train_img = (np.array(train_img))
test_img = (np.array(test_img))

train_labels_encoded = [
    0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
    for category in list(train_labels)]
test_labels_encoded = [
    0 if category == 'no_tumor' else (1 if category == 'glioma_tumor' else (2 if category == 'meningioma_tumor' else 3))
    for category in list(test_labels)]


# Shape of train and test images
# print("Shape of train: ", train_img.shape, " and shape of test: ", test_img.shape)

img_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True)

img_datagen.fit(train_img)
img_datagen.fit(test_img)

X_train, X_test, y_train, y_test = train_test_split(np.array(train_img), np.array(train_labels), test_size=0.1)
# Shape of X, y train and test samples:
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(train_img)
# Visualize some images:
plt.figure(figsize=(10, 5))
for i, j in enumerate(train_img):
    if i < 5:
        plt.subplot(1, 5, i+1)
        plt.imshow(j)
        plt.xlabel(train_labels[i])
        plt.tight_layout()
    else:
        break
plt.show()


# Visualize distributions in dataset
plt.figure(figsize=(10, 5))
lis = ['Train', 'Test']
for i, j in enumerate([train_labels, test_labels]):
    plt.subplot(1, 2, i+1)
    sns.countplot(x=j)
    plt.xlabel(lis[i])
plt.show()

