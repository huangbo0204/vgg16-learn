import random

from keras._tf_keras import keras
from keras.datasets import mnist
from keras.src import layers
from keras.src.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

num_classes = 10

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

def build_model():
    model = keras.Sequential(
        [
            keras.Input((28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    return model


model = build_model()
optimizer = RMSprop(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print("compile succeed")
n_epoch = 15
batch_size = 128


def run_model():
    # 训练
    training = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=n_epoch,
        validation_split=0.25,
        verbose=1,
    )
    # 评估
    test = model.evaluate(X_train, Y_train, verbose=1)
    return training, test


training, test = run_model()
print("model fit succeed")
print("误差:", test[0])
print("准确率:", test[1])

model.save("/huangbo/python/model/test.keras")


def show_train(train, validation):
    plt.plot(training.history[train], linestyle="-", color="b")
    plt.plot(training.history[validation], linestyle="--", color="r")
    plt.title("training history")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["training", "validation"], loc="lower right")
    plt.show()


show_train("accuracy", "val_accuracy")


def show_train1(train, validation):
    plt.plot(training.history[train], linestyle="-", color="b")
    plt.plot(training.history[validation], linestyle="--", color="r")
    plt.title("training history")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["training", "validation"], loc="upper right")
    plt.show()


show_train1("loss", "val_loss")

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


result(random.randint(0, X_test.shape[0]))
result(random.randint(0, X_test.shape[0]))
