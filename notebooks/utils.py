import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import Softmax

from src.data import METADATA, TRAIN_DATA
from src.models import BridgeResnet


def get_training_parameters(model_dir):
    with open(os.path.join(model_dir, "opts.json"), "r") as f:
        opts = json.load(f)
    return opts


def load_model(model_name, model_path, num_channels):
    model = BridgeResnet(model_name=model_name, lazy=False, num_channels=num_channels)
    model.load_state_dict(torch.load(model_path).get("model_state_dict"))
    model.eval()
    return model


def get_prediction_information(model, input_channels) -> Tuple[float, Tuple[float, float]]:
    softmax = Softmax(dim=1)
    prediction_logits = model(input_channels)
    prediction = torch.argmax(prediction_logits, -1).item()
    prediction_probabilities = softmax(prediction_logits)
    prediction_probabilities = tuple(prediction_probabilities[0].detach().numpy())
    return prediction, prediction_probabilities


def get_file_paths_from_notebook_directory():
    # correcting the file paths from ./data/[...] to ../data/[...]
    metadata = dict(METADATA)
    for country in metadata:
        md = metadata[country]
        for data_name in md:
            data = md[data_name]
            data["fp"] = os.path.join("../", data["fp"])

    train_data = dict(TRAIN_DATA)
    for version in train_data:
        for tile_size in train_data[version]:
            train_data[version][tile_size] = os.path.join(
                "../", train_data[version][tile_size])

    stats_fp = "../data/ground_truth/stats.json"
    return train_data, metadata, stats_fp


def plot_sample(sample, modalities):
    index = 0
    while index < len(sample):
        num_subplots = min(3, len(sample) - index)

        fig, axes = plt.subplots(2, num_subplots, figsize=(num_subplots * 4, 5))

        for subplot in range(num_subplots):
            plt.subplot(131 + subplot)
            plt.title(modalities[index + subplot])
            plt.imshow(sample[index + subplot])
            plt.axis('off')
        index += num_subplots
        plt.show()
