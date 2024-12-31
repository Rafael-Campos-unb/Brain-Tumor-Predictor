import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import keras_tuner
import os
import cv2
from sklearn.model_selection import train_test_split


def build_model(hp):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(kernel_size=(5, 5), filters=32,
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(kernel_size=(3, 3), filters=32,
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(kernel_size=(3, 3), filters=32,
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(kernel_size=(3, 3), filters=64,
                                activation='relu', padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(units=4, activation='softmax')
        ]
    )
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory='/BrainTumorPredictor',
    project_name='HyperparamsTuning',
)
tuner.search_space_summary()

# Prepare Data
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

# Start search
tuner.search(tf.cast(X_train, tf.float32), np.array(pd.get_dummies(y_train)), epochs=20,
             validation_data=(tf.cast(X_test, tf.float32), np.array(pd.get_dummies(y_test))))

# Get best model
models = tuner.get_best_models(num_models=2)
best_model = models[0]
print(best_model.summary)
