import pandas as pd
import numpy as np
from simple_neural_network.NeuralNetwork import NeuralNetwork
from random import shuffle


def get_data(path):
    data = pd.read_csv(path)
    labels = np.array(data['label'])
    digits = np.array(data.drop('label', axis=1)) / 255
    digits = digits.reshape(digits.shape[0], 28, 28)
    return list(zip(list(digits), list(labels)))


if __name__ == "__main__":
    train_data = get_data("data/mnist_train.csv")
    test_data = get_data("data/mnist_test.csv")
    shuffle(train_data)

    nn = NeuralNetwork()
    nn.load("model.npz")
    # nn.add_layer(28 * 28)
    # nn.add_layer(15)
    # nn.add_layer(10)

    nn.train(train_data, test_data=test_data, test=True, lr=0.5, batch_size=10, epochs=20, sleep_time=10)


