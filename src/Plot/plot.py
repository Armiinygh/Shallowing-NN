from pathlib import Path
from Config.config import cfg
import matplotlib.pyplot as plt


def plot_loss(loss, mode, foldername, filename):
    plot_directory = Path(f"src/Plot/{foldername}")
    plot_directory.mkdir(parents=True, exist_ok=True)

    if mode == "Train":
        plt.clf()
        plt.plot(loss)
        plt.title("Train loss values in each epoch step")
        labels = [i + 1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks=x, labels=labels)
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.savefig(plot_directory / filename)

    if mode == "Test":
        plt.plot(loss)
        plt.title("Test loss values in each epoch step")
        labels = [i + 1 for i in range(cfg.epochs)]
        x = list(range(cfg.epochs))
        plt.xticks(ticks=x, labels=labels)
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.gca().legend(("Train", "Test"))
        plt.savefig(plot_directory / filename)

    if mode == "Confidence":
        plt.clf()
        plt.hist(
            loss, bins=20, range=(0, 1), color="skyblue", edgecolor="black", alpha=0.7
        )
        plt.title("Confidence values in each epoch step")
        plt.xlabel("Confidence value")
        plt.ylabel("Frequency")
        plt.savefig(plot_directory / filename)
    if mode == "Correctness":
        plt.clf()
        confidences, predictions, targets = loss
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        correct = (predictions == targets).astype(float)
        plt.scatter(range(len(confidences)), confidences, c=correct, cmap="viridis")
        plt.title("Prediction Correctness vs Confidence")
        plt.xlabel("Index")
        plt.ylabel("Confidence value")
        plt.colorbar(label="Correct (1) / Incorrect (0)")
        plt.savefig(plot_directory / filename)
