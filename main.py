import sys 
sys.path.append('src')
from train.train_model import train_model
from Config.config import cfg


if __name__ == "__main__":
    # Choose which model(s) to train by commenting out uncommenting the following lines:
    train_model('shallow')
    train_model('deep')
