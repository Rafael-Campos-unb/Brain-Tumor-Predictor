from model import CNN_model
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


train_img = []
train_labels = []
path_train = 'C:/BrainTumorPredictor/Training/'
img_size = 300

for i in os.listdir(path_train):
    for j in os.listdir(path_train + i):
        train_img.append(cv2.resize(cv2.imread(path_train + i + '/' + j), (img_size, img_size)))
        train_labels.append(i)

X_train, X_test, y_train, y_test = train_test_split(np.array(train_img), np.array(train_labels), test_size=0.1)
train_img = (np.array(train_img))


cnn = CNN_model((5, 5), (3, 3), 32, 64, 'relu',
                'softmax', 'same', (2, 2), 128, 4, 'Adam',
                'categorical_crossentropy', 'accuracy')
cnn.build_model()
cnn.compile()
cnn.fit(X_train, y_train, X_test, y_test)
cnn.saved_model()

