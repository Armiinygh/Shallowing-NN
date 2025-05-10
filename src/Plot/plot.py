from Config.config import cfg
import matplotlib.pyplot as plt


def plot_loss(loss, mode, foldername , filename):
    
    if mode == "Train": 
        plt.clf()
        plt.plot(loss)
        plt.title("Train loss values in each epoch step")
        labels = [i+1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks = x , labels=labels)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.savefig(f'src/Plot/{foldername}/{filename}')
    
    if mode == "Test":
        plt.plot(loss)
        plt.title("Test loss values in each epoch step")
        labels = [i+1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks = x , labels=labels)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.gca().legend(('Train','Test'))
        plt.savefig(f'src/Plot/{foldername}/{filename}')
        
    if mode == "Confidence":
        plt.clf()
        plt.hist(loss, bins=20, range=(0, 1), color='skyblue',edgecolor='black' , alpha=0.7)
        plt.title("Confidence values in each epoch step")
        plt.xlabel('Confidence value')
        plt.ylabel('Frequency')
        plt.savefig(f'src/Plot/{foldername}/{filename}')
    if mode == "Correctness":
        plt.clf()
        confidences, predictions, targets = loss 
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        correct = (predictions == targets).astype(float)
        plt.scatter(range(len(confidences)), confidences, c=correct, cmap='viridis')
        plt.title("Prediction Correctness vs Confidence")
        plt.xlabel('Index')
        plt.ylabel('Confidence value')
        plt.colorbar(label='Correct (1) / Incorrect (0)')
        plt.savefig(f'src/Plot/{foldername}/{filename}')


