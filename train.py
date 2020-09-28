import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

from data.gen_data import gen_distance_peak_data, gen_distance_peak_data_choice

def get_model():
    model = models.Sequential()
    model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(None, 2)))
    model.add(MaxPooling1D(2))
    
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(512, 3, activation='relu', padding='same'))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='linear'))

    return model

if __name__ == '__main__':
    '''
    X, y = gen_distance_peak_data()
    X = np.swapaxes(X, 1, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    '''

    X_train, y_train = gen_distance_peak_data_choice(100000)
    X_valid, y_valid = gen_distance_peak_data(1000)

    X_train = np.swapaxes(X_train, 1, 2)
    X_valid = np.swapaxes(X_valid, 1, 2)

    model = get_model()
    model.summary()

    model.compile('adam', loss='MeanAbsoluteError')
    model.fit(X_train, y_train, batch_size=64, 
                epochs=100, validation_data=(X_valid, y_valid),
                callbacks=[
                    EarlyStopping(patience=5)
                ])