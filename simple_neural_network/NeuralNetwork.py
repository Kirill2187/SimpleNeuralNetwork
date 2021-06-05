import numpy as np

from simple_neural_network.Layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def feedforward(self, picture):
        a, z = [0] * len(self.layers), [0] * len(self.layers)
        picture = picture.reshape(picture.size, 1)
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
        return np.argmax(picture), np.max(picture)

    def add_layer(self, n):
        if len(self.layers) == 0:
            self.layers.append(Layer(n))
        else:
            self.layers.append(Layer(n, previous_layer_size=self.layers[-1].size))

    def train(self, train_data, test_data=None, epochs=10, batch_size=10, lr=1, test=False):
        for epoch in range(epochs):
            # Training
            for i in range(0, len(train_data), batch_size):
                self.__process_batch(train_data[i:i + batch_size], lr=lr)
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

    def __process_batch(self, data, lr=1):
        delta_w = [np.zeros(self.layers[i].weights.shape) for i in range(1, len(self.layers))]
        delta_b = [np.zeros(self.layers[i].biases.shape) for i in range(1, len(self.layers))]
        for example in data:
            ans = np.zeros(10)
            ans[example[1]] = 1.0
            res = self.__process_example(example[0], ans)
            for i in range(len(self.layers) - 1):
                delta_w[i] += res[0][i]
                delta_b[i] += res[1][i]
        for i in range(1, len(self.layers)):
            self.layers[i].weights += delta_w[i - 1] / len(data) * lr
            self.layers[i].biases += delta_b[i - 1] / len(data) * lr

    def __process_example(self, picture, ans):
        res, a, z = self.feedforward(picture)
        res = res.reshape(10)
        err = np.sum(np.square(ans - res))

        delta_w = [np.zeros(self.layers[i].weights.shape) for i in range(1, len(self.layers))]
        delta_b = [np.zeros(self.layers[i].biases.shape) for i in range(1, len(self.layers))]

        for layer in range(len(self.layers), 0, -1):
            pass

        return delta_w, delta_b

    @staticmethod
    def __sigmoid(arr):
        arr = np.clip(arr, -500, 500)
        return 1.0 / (1.0 + np.exp(-arr))
