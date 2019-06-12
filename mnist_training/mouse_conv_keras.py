from __future__ import print_function

import math
import os
import pickle
from time import strftime, gmtime

import cv2
import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CREATE_DATASET = True


def show_img(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def load_dataset():
    with open('mouse_dataset.pickle', 'rb') as f:
        x, y = pickle.load(f)
    return np.array(x), np.array(y)


def create_dataset():
    mouse_on_right = rgb2gray(plt.imread("mouse_dataset/img001.jpeg"))
    mouse_on_left = rgb2gray(plt.imread("mouse_dataset/img010.jpeg"))

    left_cage_empty = mouse_on_right[60:530, 60:380]
    right_cage_empty = mouse_on_left[60:530, 380:680]
    empty_cage = np.concatenate((left_cage_empty, right_cage_empty), axis=1)

    files = [filename for filename in os.listdir('mouse_dataset/') if filename.endswith('.jpeg')]

    x = []
    y = []
    for f in files:
        print(f)
        img = cv2.imread('mouse_dataset/%s' % f)

        img = rgb2gray(img)
        img = img[60:530, 60:680]

        height, width = img.shape
        height = math.ceil(height / 5)
        width = math.ceil(width / 10)
        resized_img = cv2.resize(img, (int(height), int(width)))

        mouse_location = empty_cage - img

        # filterout all values less then 0.15
        mouse_location[mouse_location < 0.15] = 0

        # compartments
        part1 = mouse_location[:, :220]
        part2 = mouse_location[:, 220:405]
        part3 = mouse_location[:, 405:]

        values = np.array([np.sum(part1), np.sum(part2), np.sum(part3)])
        mouse_compartment = np.argmax(values)
        x.append(resized_img)
        y.append(mouse_compartment)

    with open('mouse_dataset.pickle', 'wb') as f:
        pickle.dump((x, y), f)
    print('pickle created')

batch_size = 128
num_classes = 3
epochs = 20

x, y = load_dataset()
input_shape = [x.shape[1], x.shape[2], 1]
print('shape:', input_shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# channel last reshape
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(lr=0.001), metrics=['accuracy'])

date = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
callback = keras.callbacks.TensorBoard(log_dir='tensorboard/%s' % date, update_freq='batch')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    callbacks=[callback])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



