import numpy as np


class Layer:

    def __init__(self, size, previous_layer_size=0):
        self.size = size
        self.previous_layer_size = previous_layer_size
        self.is_input_layer = previous_layer_size == 0

        if not self.is_input_layer:
            self.weights = np.random.randn(size, previous_layer_size)
            self.biases = np.random.randn(size).reshape(size, 1)

    @staticmethod
    def __sigmoid(arr):
        return 1.0 / (1.0 + np.exp(-arr))

    def eval(self, result_from_prev_layer):
        if self.is_input_layer:
            raise Exception("Trying to eval input layer")
        res = self.__sigmoid(np.dot(self.weights, result_from_prev_layer) + self.biases)
        return res
