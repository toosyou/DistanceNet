import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping

from data.gen_data import gen_distance_peak_data_choice, gen_distance_peak_data
from model.distance import DistanceLayer

def get_model(input_length):
    inputs = Input(shape=(input_length, 2))
    feature = inputs
    
    feature = Conv1D(32, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)
    
    feature = Conv1D(64, 3, activation='relu', padding='same')(feature)
    feature = MaxPooling1D(2)(feature)

    output = DistanceLayer(64, 64, input_length//4, name='distance_layer')(feature)

    output = Dense(32, activation='relu')(output)
    output = Dense(1)(output)

    return Model(inputs=inputs, outputs=output)

if __name__ == '__main__':
    X_train, y_train = gen_distance_peak_data_choice(100000)
    X_valid, y_valid = gen_distance_peak_data(100000)

    X_train = np.swapaxes(X_train, 1, 2)
    X_valid = np.swapaxes(X_valid, 1, 2)

    X_train = np.pad(X_train, ((0, 0), (450, 450), (0, 0)), mode='constant', constant_values=0)
    X_valid = np.pad(X_valid, ((0, 0), (450, 450), (0, 0)), mode='constant', constant_values=0)

    model = get_model(X_train.shape[1])
    model.summary()

    model.compile('adam', loss='MeanAbsoluteError')

    model.fit(X_train, y_train, batch_size=64, 
                epochs=100, validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(patience=5)
                ])