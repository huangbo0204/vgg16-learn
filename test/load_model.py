import random

import keras.src.saving
import numpy as np
from keras.api.datasets import mnist
from keras.src.applications import imagenet_utils
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import np_utils

model = keras.src.saving.load_model("/huangbo/python/model/test.keras", compile=True)

model.summary()
num_classes = 10

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print("-------------")
X_test1 = X_test
Y_test1 = Y_test
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

prediction = model.predict(X_test)


def image_show(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap="binary")
    plt.show()


def result(i):
    image_show(X_test1[i])
    print("真实值:", Y_test1[i])
    print("预测值:", np.argmax(prediction[i]))
    names = [p[1] for p in imagenet_utils.decode_predictions(prediction)[0]]
    print(names)


result(random.randint(0, X_test.shape[0]))
result(random.randint(0, X_test.shape[0]))
