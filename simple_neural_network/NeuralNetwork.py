import numpy as np
from random import randint as rnd, random

from simple_neural_network.Layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def get_prediction(self, picture):
        picture = picture.reshape(picture.size, 1)
        for layer in self.layers[1:]:
            picture = layer.eval(picture)
        return np.argmax(picture), np.max(picture)

    def add_layer(self, n):
        if len(self.layers) == 0:
            self.layers.append(Layer(n))
        else:
            self.layers.append(Layer(n, previous_layer_size=self.layers[-1].size))

