import numpy as np
import tkinter as tk
from Blackboard import Blackboard
from simple_neural_network.NeuralNetwork import NeuralNetwork

IMAGE_SIZE = 28

nn = NeuralNetwork()
print(nn.get_prediction(0))

window = tk.Tk()
window.title("Simple Neural Network")
window.resizable(False, False)

blackboard = Blackboard(window, IMAGE_SIZE)

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

prediction_label = tk.Label(panel, text="Hmm... May be it's 5")
prediction_label.pack(ipadx=100, side=tk.RIGHT, anchor=tk.E)

window.eval('tk::PlaceWindow . center')
window.mainloop()
