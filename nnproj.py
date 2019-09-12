from keras import datasets as kdatasets
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout
from keras.models import Sequential

def train_model(X, X_2d, loss='mean_squared_error', epochs=None):
    callbacks = []

    if not epochs:
        stop = EarlyStopping(verbose=1, min_delta=0.00001, mode='min', patience=10, restore_best_weights=True)
        callbacks.append(stop)

    m = Sequential()
    m.add(Dense(256, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001),
                input_shape=(X.shape[1],)))
    m.add(Dense(512, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dense(256, activation='relu',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.add(Dense(2, activation='sigmoid',
                kernel_initializer='he_uniform',
                bias_initializer=Constant(0.0001)))
    m.compile(loss=loss, optimizer='adam')

    hist = m.fit(X, X_2d, batch_size=32, epochs=200, verbose=0, validation_split=0.05, callbacks=callbacks)

    return m, hist
