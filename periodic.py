import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from data.gen_data import gen_distance_peak_data_choice, gen_distance_peak_data
from data.gen_data import DataGenerator
from model.distance import MultiHeadDistanceLayer

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def get_model(input_length):
    inputs = Input(shape=(input_length, 2))
    feature = inputs
    
    feature = Conv1D(32, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    
    feature = Conv1D(64, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)

    feature = Conv1D(128, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)

    feature = Conv1D(256, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)

    distance_layer = MultiHeadDistanceLayer(16, 16, input_length//(2**3), name='distance_layer')
    distance_layer = tf.recompute_grad(distance_layer) # to reduce memory useages

    output = distance_layer(feature)

    output = Flatten()(output)
    output = Dense(1)(output)

    return Model(inputs=inputs, outputs=output)

if __name__ == '__main__':
    data = np.load('./data/periodic_100000.npz') # (?, 2, 5000)
    X, peaks = data['signals'], data['peaks']

    X = X.swapaxes(1, 2) # (?, 5000, 2)
    y = peaks[:, 1, :] - peaks[:, 0, :] # (?, 10)
    y = y.mean(axis=-1) # (?)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

    model = get_model(X_train.shape[1])
    model.summary()

    model.compile('adam', loss='MeanAbsoluteError')

    model.fit(X_train, y_train, batch_size=64, 
                epochs=1000, validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(patience=20),
                ])

    model.save('./periodic_model.h5')