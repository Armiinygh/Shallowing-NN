from torch import nn
from types import SimpleNamespace
# Config/config.py
class Cfg:
    def __init__(self, output_size=128, activation_function=nn.ReLU(),
                 batch_size=64, learning_rate=0.001, loss_function=nn.CrossEntropyLoss(),
                 optimization="Adam", epochs=5, save_model = True):
        self.output_size = output_size
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimization = optimization
        self.epochs = epochs
        self.save_model = save_model

cfg = Cfg()