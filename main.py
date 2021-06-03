import numpy as np
import tkinter as tk
from Blackboard import Blackboard
from simple_neural_network.NeuralNetwork import NeuralNetwork

IMAGE_SIZE = 28


def eval_network(*args):
    res = nn.get_prediction(blackboard.get_array())
    prediction_label['text'] = str(res[0])


nn = NeuralNetwork()
nn.add_layer(IMAGE_SIZE * IMAGE_SIZE)
nn.add_layer(15)
nn.add_layer(10)

window = tk.Tk()
window.title("Simple Neural Network")
window.resizable(False, False)

blackboard = Blackboard(window, IMAGE_SIZE, eval_network)

panel = tk.Frame(window)
panel.pack(side=tk.BOTTOM, anchor=tk.W)

erase_btn = tk.Button(panel, text="Clear all")
erase_btn.bind("<Button-1>", blackboard.clear)
erase_btn.pack(side=tk.LEFT)

brush_size_list = tk.Spinbox(panel, width=10,
                             from_=1, to=5,
                             command=lambda *args: blackboard.set_brush_size(int(brush_size_list.get())),
                             state="readonly",
                             textvariable=tk.DoubleVar(value=2))
brush_size_list.pack(side=tk.LEFT)

prediction_label = tk.Label(panel, text="5")
prediction_label.pack(ipadx=100, side=tk.RIGHT, anchor=tk.E)

window.eval('tk::PlaceWindow . center')
eval_network()
window.mainloop()
