import time
import pandas as pd
import numpy as np
from simple_neural_network.NeuralNetwork import NeuralNetwork
from random import shuffle, choice
from transformations import *


def get_data(path, transformations=()):
    data = pd.read_csv(path)
    labels = np.array(data['label'])
    digits = np.array(data.drop('label', axis=1)) / 255
    digits = digits.reshape(digits.shape[0], 28, 28)

    if transformations:
        transformed = np.array([choice(transformations)(i) for i in digits])
        digits = np.concatenate((digits, transformed))
        labels = np.concatenate((labels, labels))

    res = list(zip(list(digits), list(labels)))
    shuffle(res)

    print("{} loaded".format(path))

    return res


if __name__ == "__main__":
    train_data = get_data("data/mnist_train.csv", transformations=[shift_up, shift_right, shift_left, shift_down])
    test_data = get_data("data/mnist_test.csv", transformations=[shift_up, shift_right, shift_left, shift_down])

    nn = NeuralNetwork()
    nn.load("model.npz")
    # nn.add_layer(28 * 28)
    # nn.add_layer(100)
    # nn.add_layer(10)

    time.sleep(10)
    nn.train(train_data, test_data=test_data, test=True, lr=0.01, batch_size=10, epochs=30, sleep_time=30)


