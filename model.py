import keras
import pandas as pd
import tensorflow as tf
import numpy as np


class CNN_model:
    def __init__(self, conv2d_input_dim, conv2d_hidden_dim, n_filters,
                 last_conv2d_n_filters, activation, output_activation, padding, pool_dim, dense_units_in,
                 dense_units_out, optimizer, loss, metrics):
        self.model = None
        self.conv2d_input_dim = conv2d_input_dim
        self.conv2d_hidden_dim = conv2d_hidden_dim
        self.n_filters = n_filters
        self.last_conv2d_n_filters = last_conv2d_n_filters
        self.activation = activation
        self.output_activation = output_activation
        self.padding = padding
        self.pool_dim = pool_dim
        self.units_in = dense_units_in
        self.units_out = dense_units_out
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self):
        self.model = keras.Sequential(
            [
                keras.layers.Conv2D(kernel_size=self.conv2d_input_dim, filters=self.n_filters,
                                    activation=self.activation, padding=self.padding),
                keras.layers.MaxPool2D(pool_size=self.pool_dim),
                keras.layers.Conv2D(kernel_size=self.conv2d_hidden_dim, filters=self.n_filters,
                                    activation=self.activation, padding=self.padding),
                keras.layers.MaxPool2D(pool_size=self.pool_dim),
                keras.layers.Conv2D(kernel_size=self.conv2d_hidden_dim, filters=self.n_filters,
                                    activation=self.activation, padding=self.padding),
                keras.layers.MaxPool2D(pool_size=self.pool_dim),
                keras.layers.Conv2D(kernel_size=self.conv2d_hidden_dim, filters=self.last_conv2d_n_filters,
                                    activation=self.activation, padding=self.padding),
                keras.layers.MaxPool2D(pool_size=self.pool_dim),
                keras.layers.Flatten(),
                keras.layers.Dense(units=self.units_in, activation=self.activation),
                keras.layers.Dropout(rate=0.5),
                keras.layers.Dense(units=self.units_out, activation=self.output_activation)
            ]
        )

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics]
                           )

    def fit(self, X_train, X_train_labels, X_val, X_val_labels):
        self.model.fit(tf.cast(X_train, tf.float32), np.array(pd.get_dummies(X_train_labels)),
                       validation_data=(tf.cast(X_val, tf.float32), np.array(pd.get_dummies(X_val_labels))),
                       validation_split=0.1,
                       epochs=20, verbose=1, batch_size=32)

    def saved_model(self):
        self.model.save('model.keras')
