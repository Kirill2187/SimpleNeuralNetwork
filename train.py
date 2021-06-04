import pandas as pd
import numpy as np
from simple_neural_network.NeuralNetwork import NeuralNetwork

EPOCHS = 10


def get_data(path):
    data = pd.read_csv(path)
    labels = np.array(data['label'])
    digits = np.array(data.drop('label', axis=1)) / 255
    digits = digits.reshape(digits.shape[0], 28, 28)
    return list(zip(list(digits), list(labels)))


train_data = get_data("data/mnist_train.csv")
test_data = get_data("data/mnist_test.csv")

nn = NeuralNetwork()
nn.add_layer(28 * 28)
nn.add_layer(15)
nn.add_layer(10)

for epoch in range(EPOCHS):
    # Training
    print("-" * 30)
    print("Epoch {} finished".format(epoch + 1))

    # Testing
    cnt = 0
    for i in range(len(test_data)):
        ans = nn.get_prediction(test_data[i][0])[0]
        if ans == test_data[i][1]:
            cnt += 1
    print("Accuracy: {} / {}".format(cnt, len(test_data)))


