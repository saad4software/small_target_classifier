from get_data import get_data_pairs as get_data
import numpy as np
from sklearn.model_selection import train_test_split
from model_lstm import get_model
from tensorflow import keras
import tensorflow as tf
import datetime

dataset_path = "dataset"

airplane_data1, y_airplane1 = get_data(f"{dataset_path}/vis/target/airplane/*_1.png", class_num=0)
bird_data1, y_bird1 = get_data(f"{dataset_path}/vis/target/bird/*_1.png", class_num=0)
noise_data1, y_noise1 = get_data(f"{dataset_path}/vis/noise/*_1.png", class_num=1)

airplane_data2, y_airplane2 = get_data(f"{dataset_path}/ir/target/airplane/*_1.png", class_num=0)
bird_data2, y_bird2 = get_data(f"{dataset_path}/ir/target/bird/*_1.png", class_num=0)
noise_data2, y_noise2 = get_data(f"{dataset_path}/ir/noise/*_1.png", class_num=1)


train_data =  airplane_data1 + noise_data1 + bird_data1 + airplane_data2 + noise_data2 + bird_data2
y_data = y_airplane1 + y_noise1 + y_bird1 + y_airplane2 + y_noise2 + y_bird2

x_train = np.array(train_data)
y_train = np.array(y_data)

# shuffle the data

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)
print(x_test.shape)


SIZE = (20, 20)
CHANNELS = 3
NBFRAME = 2
BS = 8

classes = ["target", "noise"] 

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
# model = action_model(INSHAPE, 2)
model = get_model()
optimizer = keras.optimizers.Adam(0.001)
model.compile(
    optimizer,
    'sparse_categorical_crossentropy',
    metrics=['acc']
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=25, 
    batch_size=256,
    callbacks=[tensorboard_callback]

)

model.save("classifier_lstm.keras")
