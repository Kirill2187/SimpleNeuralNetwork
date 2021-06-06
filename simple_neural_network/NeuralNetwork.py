import numpy
import numpy as np
from random import shuffle
from time import sleep

from simple_neural_network.Layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def feedforward(self, picture):
        a, z = [numpy.array([0])] * len(self.layers), [numpy.array([0])] * len(self.layers)
        picture = picture.reshape(picture.size)
        a[0] = picture
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            picture = np.dot(layer.weights, picture) + layer.biases
            z[i] = picture
            picture = self.__sigmoid(picture)
            a[i] = picture

        return picture, a, z

    def get_prediction(self, picture):
        picture = self.feedforward(picture)[0]
        picture = picture / np.sum(picture)
        return np.argmax(picture), np.max(picture)

    def add_layer(self, n):
        if len(self.layers) == 0:
            self.layers.append(Layer(n))
        else:
            self.layers.append(Layer(n, previous_layer_size=self.layers[-1].size))

    def train(self, train_data, test_data=None, epochs=10, batch_size=10, lr=1,
              regularization=True, regularization_coefficient=5.0,
              test=False, print_epoch_progress=False, save_best=True, filename="model", sleep_time=0):
        best_accuracy = 0
        for epoch in range(epochs):
            shuffle(train_data)

            # Training
            for i in range(0, len(train_data), batch_size):
                if print_epoch_progress:
                    print("{} / {} done".format(i, len(train_data)))
                self.__process_batch(train_data[i:i + batch_size], lr=lr,
                                     regularization=regularization,
                                     regularization_coefficient=regularization_coefficient / len(train_data))
            print("-" * 30)
            print("Epoch {} finished".format(epoch + 1))

            # Testing
            if test:
                cnt = 0
                for i in range(len(test_data)):
                    ans = self.get_prediction(test_data[i][0])[0]
                    if ans == test_data[i][1]:
                        cnt += 1
                print("Accuracy: {} / {}".format(cnt, len(test_data)))
                if cnt > best_accuracy and save_best:
                    best_accuracy = cnt
                    self.save(filename)

            print("Sleep for {} seconds...".format(sleep_time))
            sleep(sleep_time)

    def __process_batch(self, data, lr=1, regularization=True, regularization_coefficient=5.0):
        delta_w = [np.zeros(self.layers[i].weights.shape) for i in range(1, len(self.layers))]
        delta_b = [np.zeros(self.layers[i].biases.shape) for i in range(1, len(self.layers))]
        for example in data:
            ans = np.zeros(self.layers[-1].size)
            ans[example[1]] = 1.0
            res = self.__process_example(example[0], ans)
            for i in range(len(self.layers) - 1):
                delta_w[i] += res[0][i]
                delta_b[i] += res[1][i]
        for i in range(1, len(self.layers)):
            self.layers[i].weights -= delta_w[i - 1] / len(data) * lr
            if regularization:
                self.layers[i].weights -= self.layers[i].weights * (lr * regularization_coefficient)
            self.layers[i].biases -= delta_b[i - 1] / len(data) * lr

    def __process_example(self, picture, ans):
        res, a, z = self.feedforward(picture)
        res = res.reshape(self.layers[-1].size)

        delta_w = [np.zeros(self.layers[i].weights.shape) for i in range(1, len(self.layers))]
        delta_b = [np.zeros(self.layers[i].biases.shape) for i in range(1, len(self.layers))]

        partial_z = (res - ans)

        for num in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[num]
            delta_b[num - 1] = partial_z
            delta_w[num - 1] = np.outer(partial_z, a[num - 1])

            if num != 1:
                partial_z = np.multiply(np.dot(layer.weights.T, partial_z), self.__sigmoid_prime(z[num - 1]))

        return delta_w, delta_b

    def save(self, filename):
        print("Saving to file...")
        layer_dict = {}
        for i in range(len(self.layers)):
            layer_dict["layer_{}_weights".format(i)] = self.layers[i].weights if i > 0 else np.array([])
            layer_dict["layer_{}_biases".format(i)] = self.layers[i].biases if i > 0 else np.array([])
        np.savez(filename, **layer_dict)
        print("Saving completed")

    def load(self, filename):
        print("Loading from file...")
        self.layers.clear()
        model = np.load(filename)
        size = len(model.files) // 2
        for i in range(size):
            name = "layer_" + str(i) + "_"
            self.add_layer(model[name + "biases"].size)
            if i == 0:
                continue
            self.layers[-1].weights = model[name + "weights"]
            self.layers[-1].biases = model[name + "biases"]
        print("Loading completed")

    @staticmethod
    def __sigmoid(arr):
        arr = np.clip(arr, -500, 500)
        return 1.0 / (1.0 + np.exp(-arr))

    @staticmethod
    def __sigmoid_prime(arr):
        sig = NeuralNetwork.__sigmoid(arr)
        return np.multiply(sig, 1 - sig)
