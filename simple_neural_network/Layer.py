import numpy as np


class Layer:

    def __init__(self, size, previous_layer_size=0):
        self.size = size
        self.previous_layer_size = previous_layer_size
        self.is_input_layer = previous_layer_size == 0

        if not self.is_input_layer:
            self.weights = np.random.randn(size, previous_layer_size)
            self.biases = np.random.randn(size)
