import sys 
sys.path.append('src')
from Shallow_nn.Shallow_nn import train_shallow_model
from Config.config import cfg
from Deep_nn.Deep_nn import train_deep_model

if __name__ == "__main__" :
    train_shallow_model()
    train_deep_model()