import torch 
from torch import nn 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import ssl
from Config.config import cfg
from Plot.plot import plot_loss

ssl._create_default_https_context = ssl._create_unverified_context

train_loss = []
test_loss = []


class FeedForwadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(
                28*28, cfg.output_size
            ), 
            cfg.activation_function,
            nn.Linear(cfg.output_size, 10)
        )
        self.softmax = nn.Softmax(dim = 1)
        
    
    def forward (self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        return logits
    

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

    train_loss.append(loss.item())
    print(f"Loss : {loss.item()}")

def train(model , data_loader, loss_fn, optimizer, device, epochs) :
    for i in range(epochs):
        print(f"Epoch :{i+1}")
        train_one_epoch(model , data_loader, loss_fn, optimizer, device)
        print("-" * 10)
    plot_loss(train_loss , "Train")
    print("Training is complete!")     


def train_model():
    train_data , test_data = download_mnist_datasets()
    print("MNIST dataset has been downloaded")



    train_data_loader = DataLoader(train_data , batch_size = cfg.batch_size)
    test_data_loader = DataLoader(test_data , batch_size = cfg.batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = FeedForwadNet().to(device=device)
    
    loss_fn = cfg.loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.learning_rate)
    
    train(model, train_data_loader, loss_fn, optimizer , device ,cfg.epochs)
    
    #Evaluation 
    correct = 0
    total = 0
    i = 0
    model.eval()
    with torch.no_grad():
            
        for images, targets in test_data_loader:
            if i == cfg.epochs:
                break
            images, targets = images.to(device), targets.to(device)
            targets = targets.long()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, predictions  = torch.max(probs, dim=1)
            # Use outputs (not predictions) for loss calculation
            test_loss.append(loss_fn(outputs, targets).item())
            _, prediction = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (prediction == targets).sum().item()
            i += 1
    plot_loss(test_loss, "Test")
    plot_loss(confidence, "Confidence")
    data = confidence, predictions, targets
    plot_loss(data, "Correctness")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    
    file_path = "feedforwardnet.pth"
    torch.save(model.state_dict(), file_path)
    
    
    print(f"Model has been trained and stored at {file_path}")
    
    
    
    #TODO separate each module to a class 
    #TODO plot the loss functions and every other parameter that make sense using different values and compare it to the results of the deep neural networks 
    
    