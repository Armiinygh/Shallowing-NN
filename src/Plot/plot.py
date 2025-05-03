from Config.config import cfg
import matplotlib.pyplot as plt


def plot_loss(loss):
    plt.plot(loss)
    plt.title("Loss values in each epoch step")
    labels = [i+1 for i in range(cfg.epochs)]
    x = list(range(cfg.epochs))
    plt.xticks(ticks = x , labels=labels)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.savefig('Loss_function.png')
    


