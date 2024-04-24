from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, InputLayer
from keras.layers import TimeDistributed, GRU, Dense, Dropout, LSTM


def build_convnet_old(shape=(20, 20, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(InputLayer(shape))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

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


def get_model(shape=(2, 20, 20, 3), nbout=2):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet()
    
    print(f"number of outputs are {nbout}")
    
    # then create our final model
    model = keras.Sequential()
    model.add(TimeDistributed(convnet, input_shape=shape))

    model.add(LSTM(64))
    model.add(Dense(nbout, activation='softmax'))

    return model



