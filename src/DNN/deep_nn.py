import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class DeepNet(nn.Module):

    # building the DNN
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512, bias = True), 
            nn.ReLU(), 
            nn.Linear(512, 128, bias = True), 
            nn.ReLU(), 
            nn.Linear(128, 256, bias = True), 
            nn.ReLU(), 
            nn.Linear(256, 128, bias = True), 
            nn.ReLU(), 
            nn.Linear(128, 512, bias = True), 
            nn.ReLU(), 
            nn.Linear(512, 10, bias = True)
        )
        # calculating loss while also turning logits into probabilities
        self.loss = nn.CrossEntropyLoss()

    # defining how data moves through NN    
    def forward(self, data):
        data = self.flatten(data)
        logits = self.layers(data)
        return logits
    
    # defining output classes
    labels = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

    # defining the computation of loss
    def compute_loss(self, logits, labels):
        return self.loss(logits, labels)
    
    # loading the data
    def data_download():
        data = datasets.MNIST(
            root = "./data",
            download = True,
            train = True, 
            transform = ToTensor()
        )

        test_data = datasets.MNIST(
            root = "./data",
            download = True,
            train = False, 
            transform = ToTensor()
        )

        # splitting the data into batches for decreasing computational demand duwing training
        train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        return train_loader

# setting model and optimizer
model = DeepNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_loader = DeepNet.data_download()

# setting epochs
n_epochs = 10

for epoch in range(n_epochs):
    for batch in train_loader:
        X_batch, y_batch = batch
        y_pred = model(X_batch)
        loss = model.compute_loss(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # printing current epoch starting from 1 not o, printing loss up to 4 decimals
    print(f'Finished epoch {epoch + 1}, latest loss: {loss.item():.4f}')

#EVALUATION
model.eval()  

# Plot results with mathplotlib
# MSE & accuracy & correctness & confidency (probabilities - logits after using softmax)