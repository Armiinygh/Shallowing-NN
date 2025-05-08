"""
Plotting utility for training visualization: loss curves, confidence histograms, and prediction correctness scatter plots.
"""

from Config.config import cfg
import matplotlib.pyplot as plt

def plot_loss(data, mode, model):
    """
    Visualizes training results using different kinds of plots.

    Args:
        data (list or tuple): 
            - For 'Train' or 'Test', a list/array of loss per epoch.
            - For 'Confidence', a 1D list/array/tensor of confidence scores (0 to 1).
            - For 'Correctness', a tuple (confidences, predictions, targets) as tensors.
        mode (str): 
            - 'Train' for training loss curve.
            - 'Test' for test loss curve.
            - 'Confidence' for histogram of confidence scores.
            - 'Correctness' for scatter plot of correctness vs. confidence.
        model (str): Name of the model ("Deep", "Shallow", etc).
    """
    plt.clf()
    save_folder = "src/Plot" 
    if mode == "Train":
        plt.plot(data, label="Train Loss", color='blue')
        plt.title(f"{model}: Train loss per epoch")
        labels = [i+1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks=x, labels=labels)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{model.lower()}_train_loss.png')
    
    elif mode == "Test":
        plt.plot(data, label="Test Loss", color='red')
        plt.title(f"{model}: Test loss per epoch")
        labels = [i+1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks=x, labels=labels)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{model.lower()}_test_loss.png')
        
    elif mode == "Confidence":
        plt.hist(data, bins=20, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7, label='Confidence')
        plt.title(f"{model}: Confidence values distribution")
        plt.xlabel('Confidence value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{model.lower()}_confidence.png')

    elif mode == "Correctness":
        confidences, predictions, targets = data 
        if hasattr(confidences, "cpu"):
            confidences = confidences.cpu().numpy()
        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()
        if hasattr(targets, "cpu"):
            targets = targets.cpu().numpy()
        correct = (predictions == targets).astype(float)
        scatter = plt.scatter(range(len(confidences)), confidences, c=correct, cmap='viridis', label='Prediction')
        plt.title(f"{model}: Prediction correctness vs. confidence")
        plt.xlabel('Index')
        plt.ylabel('Confidence value')
        plt.colorbar(scatter, label='Correct (1) / Incorrect (0)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{model.lower()}_correctness.png')

    plt.close()