"""
Deep Neural Network (DNN) model definition.

Implements a multi-layer perceptron with 6 hidden layers.
"""

import torch.nn as nn

class DeepNet(nn.Module):
    """
    Deep neural network model for MNIST classification.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, data):
        """
        Forward pass for the model.

        Args:
            data (Tensor): Batch of images.

        Returns:
            Tensor: Logits for each class.
        """
        data = self.flatten(data)
        return self.layers(data)