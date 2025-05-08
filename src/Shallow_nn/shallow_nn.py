"""
Shallow feedforward neural network definition for MNIST classification.
"""

import torch.nn as nn
from Config.config import cfg

class FeedForwardNet(nn.Module):
    """
    Shallow neural network with one hidden layer and configurable activation.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, cfg.output_size),
            cfg.activation_function,
            nn.Linear(cfg.output_size, 10)
        )

    def forward(self, input_data):
        """
        Forward pass for the model.

        Args:
            input_data (Tensor): Batch of images.

        Returns:
            Tensor: Logits for each class.
        """
        return self.dense_layers(self.flatten(input_data))