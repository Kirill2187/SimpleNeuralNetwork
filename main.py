from random import randint
import numpy as np
import tkinter as tk
from Blackboard import Blackboard
from simple_neural_network.NeuralNetwork import NeuralNetwork
from train import get_data

IMAGE_SIZE = 28


def eval_network(*args):
    res = nn.get_prediction(blackboard.get_array())
    prediction_label['text'] = "{} with a probability of {:.2f}".format(res[0], res[1])


nn = NeuralNetwork()
nn.load("models/model97.npz")
test_data = get_data("data/mnist_test.csv")

window = tk.Tk()
window.title("Simple Neural Network")
window.resizable(False, False)

blackboard = Blackboard(window, IMAGE_SIZE, eval_network)

panel = tk.Frame(window)
panel.pack(side=tk.BOTTOM, anchor=tk.W)

erase_btn = tk.Button(panel, text="Clear all")
erase_btn.bind("<Button-1>", blackboard.clear)
erase_btn.pack(side=tk.LEFT)

show_example = tk.Button(panel, text="Load example")
show_example.bind("<Button-1>", lambda *args: blackboard.show(test_data[randint(0, 9999)][0]))
show_example.pack(side=tk.LEFT)

brush_size_list = tk.Spinbox(panel, width=10,
                             from_=1, to=4,
                             command=lambda *args: blackboard.set_brush_size(int(brush_size_list.get())),
                             state="readonly",
                             textvariable=tk.DoubleVar(value=2))
brush_size_list.pack(side=tk.LEFT)

prediction_label = tk.Label(panel, text="5")
prediction_label.pack(ipadx=25, side=tk.RIGHT, anchor=tk.E)

window.eval('tk::PlaceWindow . center')
eval_network()

window.mainloop()
