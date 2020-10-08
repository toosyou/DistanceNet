import numpy as np
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

    output = MultiHeadDistanceLayer(4, 16, input_length//(2**3), name='distance_layer')(feature)

    output = Flatten()(output)
    output = Dense(1)(output)

    return Model(inputs=inputs, outputs=output)

if __name__ == '__main__':
    '''
    X_train, y_train = gen_distance_peak_data_choice(100000)
    X_valid, y_valid = gen_distance_peak_data(100000)

    X_train = np.swapaxes(X_train, 1, 2)
    X_valid = np.swapaxes(X_valid, 1, 2)

    X_train = np.pad(X_train, ((0, 0), (450, 450), (0, 0)), mode='constant', constant_values=0)
    X_valid = np.pad(X_valid, ((0, 0), (450, 450), (0, 0)), mode='constant', constant_values=0)
    '''

    g = DataGenerator(num_data=10000, channel=2, signal_length=1000, padding_length=1000)
    g.addPeakShape(["triangle", "square"])
    X_train, y_train = g.generate(noisy_peak_num=4)
    X_valid, y_valid = g.generate(noisy_peak_num=4)

    X_train = np.swapaxes(X_train, 1, 2)
    X_valid = np.swapaxes(X_valid, 1, 2)

    model = get_model(X_train.shape[1])
    model.summary()

    model.compile('adam', loss='MeanAbsoluteError')

    model.fit(X_train, y_train, batch_size=64, 
                epochs=1000, validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(patience=20)
                ])