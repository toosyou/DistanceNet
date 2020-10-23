import os
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data.gen_data import gen_distance_peak_data_choice, gen_distance_peak_data
from data.gen_data import DataGenerator
from model.distance import MultiHeadDistanceLayer

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

class PriorPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        distance_layer = self.model.get_layer('distance_layer')
        tf.print(distance_layer.prior_mean, tf.exp(distance_layer.log_prior_std), summarize=-1)

def get_model(input_length):
    inputs = Input(shape=(input_length, 2))
    feature = inputs
    
    feature = Conv1D(32, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    feature = BatchNormalization()(feature)
    
    feature = Conv1D(64, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    feature = BatchNormalization()(feature)

    feature = Conv1D(128, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    feature = BatchNormalization()(feature)

    feature = Conv1D(128, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    feature = BatchNormalization()(feature)

    distance_layer = MultiHeadDistanceLayer(16, 16, input_length//(2**3), name='distance_layer')
    distance_layer = tf.recompute_grad(distance_layer) # to reduce memory useages

    output = distance_layer(feature)

    output = Flatten()(output)
    # output = Dense(2, activation='softmax')(output)
    output = Dense(1)(output)

    return Model(inputs=inputs, outputs=output)

def distance_regression():
    data = np.load('./data/periodic_100000.npz') # (?, 2, 5000)
    X, peaks = data['signals'], data['peaks']

    X = X.swapaxes(1, 2) # (?, 5000, 2)
    y = peaks[:, 1, :] - peaks[:, 0, :] # (?, 10)
    y = y.mean(axis=-1) # (?)
    return X, y

def abnormal_detection():
    X, y = distance_regression()

    y = (y > 80).astype(int)
    y = tf.keras.utils.to_categorical(y, num_classes=2, dtype='int')

    return X, y

if __name__ == '__main__':
    X, y = distance_regression()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

    model = get_model(X_train.shape[1])
    model.summary()

    # model.compile('adam', loss='CategoricalCrossentropy',
    #                 metrics='acc')
    model.compile('adam', loss='MeanAbsoluteError', metrics=['MeanAbsoluteError'])

    model.fit(X_train, y_train, batch_size=64, 
                epochs=1000, validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(patience=2),
                    PriorPrinter(),
                    ModelCheckpoint('/tmp/periodic_models', save_best_only=True)
                ])