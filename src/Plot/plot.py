from pathlib import Path
from Config.config import cfg
import matplotlib.pyplot as plt
import torch


def plot_loss(loss, mode, folderpath, filename):
    plot_directory = folderpath.joinpath(Path("Plot"))
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


def saveWeigths_PlotAll(plottables, modeldict, modelname):
    """saves the modelweights and creates plots under src/Results/modelname
    plottables dict with key: plotname and value: list of plottable items see plot_loss
    """
    results_directory = Path(f"src/Results/{modelname}")
    results_directory.mkdir(parents=True, exist_ok=True)
    for key, value in plottables.items():
        plot_loss(value, key, results_directory, f"{key}.png")
    weights_directory = results_directory.joinpath("Weights")
    weights_directory.mkdir(parents=True, exist_ok=True)
    torch.save(modeldict, f"{weights_directory.__str__()}/{modelname}.pth")
