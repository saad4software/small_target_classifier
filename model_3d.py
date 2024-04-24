from tensorflow import keras
from keras.layers import Conv2D, Conv3D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, GlobalMaxPool3D, InputLayer
from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM


def build_convnet(width=20, height=20, depth=2):
    momentum = .9
    model = keras.Sequential()
    model.add(keras.Input((depth, width, height, 3)))
    model.add(Conv3D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv3D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool3D())
    return model


def get_model():

    convnet = build_convnet()
    
    model = keras.Sequential()
    model.add(convnet)
    model.add(Dense(2, activation='softmax'))
    return model



