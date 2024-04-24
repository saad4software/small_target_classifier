from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, InputLayer
from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM


def build_convnet(shape=(20, 20, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(InputLayer(shape))
    # model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    model.add(Dropout(0.3))
    return model


def get_model(shape=(20, 20, 3), nbout=2):
    convnet = build_convnet()

    model = keras.Sequential()
    model.add(convnet)

    model.add(Dense(nbout, activation='softmax'))

    return model



