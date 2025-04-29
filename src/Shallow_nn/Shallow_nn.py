import torch 
from torch import nn 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.1

class FeedForwadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(
                28*28, 512
            ), 
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim = 1)
        #TODO add model confidency 
        
    
    def forward (self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        return self.softmax(logits)
    

def download_mnist_datasets():
    
    train_data = datasets.MNIST(
        root="./data",
        download=True,
        train=True, 
        transform=ToTensor()
        )
    
    test_data = datasets.MNIST(
        root="./data",
        download=True,
        train=False, 
        transform=ToTensor()
        )
    return train_data, test_data

def train_one_epoch(model , data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device) , targets.to(device)
        
        
        predictions= model(inputs)
        loss = loss_fn(predictions , targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss : {loss.item()}")

def train(model , data_loader, loss_fn, optimizer, device, epochs) :
    for i in range(epochs):
        print(f"Epoch :{i+1}")
        train_one_epoch(model , data_loader, loss_fn, optimizer, device)
        print("-" * 10)
    print("Training is complete!")
     


if __name__ == "__main__":
    train_data , _ = download_mnist_datasets()
    print("MNIST dataset has been downloaded")



    train_data_loader = DataLoader(train_data , batch_size = BATCH_SIZE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    feed_forward_net = FeedForwadNet().to(device=device)
    
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr = LEARNING_RATE)
    
    
    
    
    train(feed_forward_net, train_data_loader, loss_fn, optimizer , device ,EPOCHS)
    
    file_path = "feedforwardnet.pth"
    torch.save(feed_forward_net.state_dict(), file_path)
    
    
    print(f"Model has been trained and stored at {file_path}")
    
    
    
    #TODO separate each module to a class 
    #BUG the weights should update after each iteration, Curretnly they can not be updated!
    
    
    
    