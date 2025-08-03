import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import ssl
from Config.config import Cfg
from Plot.plot import plot_loss
import torch.nn.init as init

ssl._create_default_https_context = ssl._create_unverified_context

train_loss = []
test_loss = []


class FeedForwadNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.flatten = nn.Flatten()
        input_layer = nn.Linear(28 * 28, 512)
        init.xavier_uniform_(input_layer.weight)
        self.dense_layers = nn.Sequential(
            input_layer,
            cfg.activation_function,
            nn.Linear(512, 256),
            cfg.activation_function,
            nn.Linear(256, 128),
            cfg.activation_function,
            nn.Linear(128, cfg.output_size),
            cfg.activation_function,
            nn.Linear(cfg.output_size, 10),
        )

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        return logits

def download_mnist_datasets():

    train_data = datasets.MNIST(
        root="./data", download=True, train=True, transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="./data", download=True, train=False, transform=ToTensor()
    )
    return train_data, test_data


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss.append(loss.item())
    print(f"Loss : {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    num_epochs = int(round(epochs))
    for i in range(num_epochs):
        print(f"Epoch :{i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-" * 10)
    plot_loss(train_loss, "Train", "deep-nn", "Train_loss_deep.png")
    print("Training is complete!")


def train_deep_model(cfg):
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    loss_function = cfg.loss_function
    optimization = cfg.optimization
    epochs = cfg.epochs
    train_data, test_data = download_mnist_datasets()
    print("MNIST dataset has been downloaded")

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FeedForwadNet(cfg).to(device=device)

    loss_fn = loss_function
    if optimization == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimization == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9
        )

    train(model, train_data_loader, loss_fn, optimizer, device, epochs)

    # Evaluation
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
            confidence, predictions = torch.max(probs, dim=1)
            # Use outputs (not predictions) for loss calculation
            test_loss.append(loss_fn(outputs, targets).item())
            _, prediction = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (prediction == targets).sum().item()
            i += 1
    plot_loss(test_loss, "Test", "deep-nn", "Test_loss_deep.png")
    plot_loss(confidence, "Confidence", "deep-nn", "Confidence_deep.png")
    data = confidence, predictions, targets
    plot_loss(data, "Correctness", "deep-nn", "Correctness_deep.png")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    file_path = "feedforwardnet.pth"
    torch.save(model.state_dict(), file_path)

    print(f"Model has been trained and stored at {file_path}")

    return 100 * correct / total



def bo_train_deep_model(learning_rate, batch_size, output_size, loss_function_idx,activation_function_idx, epochs):
    # Create a cfg object with the values provided by the optimizer
    # Use a consistent activation function and other non-optimized parameters
    loss_functions_map = [
    nn.CrossEntropyLoss(),
    nn.NLLLoss() 
    ]
    
    selected_index = int(round(loss_function_idx))
    selected_loss_fn = loss_functions_map[selected_index]
    
        
    activation_functions_map = [
    nn.ReLU(),
    nn.Tanh(),
    nn.Sigmoid()
    ]
    
    selected_index = int(round(activation_function_idx))
    selected_activation_fn = activation_functions_map[selected_index]
    
    
    cfg = Cfg(
        output_size=int(round(output_size)),  
        learning_rate=learning_rate,
        batch_size=int(round(batch_size)),     
        activation_function= selected_activation_fn,         
        loss_function=selected_loss_fn,  
        optimization="Adam",                   
        epochs=epochs
    )

# The function `train_shallow_model` is called with the newly created cfg object.
# It should return the value you want to maximize, e.g., accuracy.
    accuracy = train_deep_model(cfg)
    return accuracy