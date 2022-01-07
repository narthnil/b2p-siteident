import os
from typing import List

import matplotlib.pyplot as plt


def plot_single_statistic(epochs: List[int], statistics: dict, file_path: str):
    _type = file_path.replace(".png", "")
    plt.figure(figsize=(10, 10))
    plt.title(f"{_type} over training")
    plt.xlabel("Number of epochs")
    plt.ylabel(_type)
    plt.plot(epochs, statistics["train"], label="train")
    plt.plot(epochs, statistics["validation"], label="validation")
    plt.plot(epochs, statistics["test"], label="test")
    plt.legend()
    plt.savefig(file_path)
    plt.close()


def plot_and_save_training_statistics(save_dir: str, accuracies: dict, losses: dict):
    num_epochs = len(accuracies["train"])
    epochs = list(range(1, num_epochs + 1))
    accuracy_fp = os.path.join(save_dir, "accuracies.png")
    loss_fp = os.path.join(save_dir, "loss.png")
    plot_single_statistic(epochs, accuracies, accuracy_fp)
    plot_single_statistic(epochs, losses, loss_fp)
