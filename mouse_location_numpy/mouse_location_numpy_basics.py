import os

import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def show_img(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


# time in seconds in all 3 compartments
compartments_time = [0, 0, 0]

mouse_on_right = rgb2gray(plt.imread("mouse_dataset/img001.jpeg"))
mouse_on_left = rgb2gray(plt.imread("mouse_dataset/img010.jpeg"))

left_cage_empty = mouse_on_right[60:530, 60:380]
right_cage_empty = mouse_on_left[60:530, 380:680]
empty_cage = np.concatenate((left_cage_empty, right_cage_empty), axis=1)

files = []
for file in os.listdir('mouse_dataset/'):
    if file.endswith('.jpeg'):
        files.append(file)

for f in files:
    img = plt.imread('mouse_dataset/%s' % f)
    img = rgb2gray(img)
    img = img[60:530, 60:680]
    mouse_location = empty_cage - img

    # filterout all values less then 0.15
    mouse_location[mouse_location < 0.15] = 0

    # compartments
    part1 = mouse_location[:, :220]
    part2 = mouse_location[:, 220:405]
    part3 = mouse_location[:, 405:]

    values = np.array([np.sum(part1), np.sum(part2), np.sum(part3)])
    mouse_compartment = np.argmax(values)

    print(mouse_compartment+1)
    show_img(img)
    compartments_time[mouse_compartment] += 1

print(compartments_time)
