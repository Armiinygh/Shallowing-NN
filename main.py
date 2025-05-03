import sys 
sys.path.append('src')
from Shallow_nn.Shallow_nn import train_model
from Config.config import cfg


if __name__ == "__main__" :
    train_model()