"""
Code adjusted from https://pytorch.org/tutorials/beginner/...
.../finetuning_torchvision_models_tutorial.html
"""

import argparse
import glob
import json
import shutil
import time

from os import path, makedirs

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, models, transforms

NUM_LAYERS = {
    "resnet18": 52,
    "resnet50": 126,
    "wide_resnet50_2": 126,
    "efficientnet_v2_s": 485,
    "efficientnet_v2_m": 697,
}

INPUT_SIZE = {
    "v1_b2p_rgb_large_590_320_jpg": 32,
    "v1_b2p_rgb_large_1150_600_jpg": 60,
    "v1_b2p_rgb_large_2350_1200_jpg": 120,
    "v2_b2p_rgb_large_590_320_jpg": 32,
    "v2_b2p_rgb_large_1150_600_jpg": 60,
    "v2_b2p_rgb_large_2350_1200_jpg": 120,
}


def train_model(model, dataloaders, criterion, optimizer, lr_scheduler,
                best_fp, last_fp, num_epochs=100):
    start_time = time.time()

    history = [
        ["train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"]]
    best_acc = 0.0

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an
                    # auxiliary output. In train mode we calculate the loss by
                    # summing the final output and the auxiliary output but in
                    # testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)
            epoch_acc = epoch_acc.item()

            print('Epoch {} [{}] Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "train":
                history.append([
                    round(epoch_loss, 4), round(epoch_acc * 100, 4)])
            else:
                history[-1] += [
                    round(epoch_loss, 4), round(epoch_acc * 100, 4)]
                lr_scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    history[-1].append(1)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'val_loss': epoch_loss,
                        "epoch": epoch
                    }, best_fp)
                else:
                    history[-1].append(0)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'val_loss': epoch_loss,
        "epoch": epoch
    }, last_fp)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.2f}%'.format(best_acc * 100))

    # load best model weights
    checkpoint = torch.load(best_fp)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, history


def test_best_model(model, criterion, dataloaders):
    model.eval()
    stats = {}
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    format_str = "Test [{}] Loss: {:.4f} Acc: {:.4f} F1(Weighted): {:.4f}"
    with torch.no_grad():
        for name, dataloader in dataloaders.items():
            # Iterate over data.
            running_loss = 0.0
            running_corrects = 0
            y_pred = []
            y_true = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_true += labels.cpu().numpy().tolist()
                y_pred += preds.cpu().numpy().tolist()

            # statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = (
                running_corrects.double() / len(dataloader.dataset)).item()
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
            stats[name + "_loss"] = round(epoch_loss, 4)
            stats[name + "_acc"] = round(epoch_acc * 100, 4)
            stats[name + "_weighted_f1"] = round(weighted_f1, 4)

            print(format_str.format(
                name, epoch_loss, epoch_acc * 100, weighted_f1))
    return stats


def initialize_model(model_name, num_classes, use_last_n_layers=1,
                     use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each
    # of these variables is model specific
    if use_pretrained:
        if model_name.startswith("efficientnet") or model_name == "resnet18":
            weights = "IMAGENET1K_V1"
        else:
            weights = "IMAGENET1K_V2"
    else:
        weights = None
    model = getattr(models, model_name)(weights=weights)
    set_parameter_requires_grad(model, model_name, use_last_n_layers)
    if model_name.startswith("efficientnet"):
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes))
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def count_num_layers(m: nn.Module):
    stack = list(m.children())
    num_layers = 0
    while stack:
        next_elem = stack.pop(0)
        children = list(next_elem.children())
        if len(children) == 0:
            num_layers += 1
        else:
            stack = children + stack
    return num_layers


def set_parameter_requires_grad(model, model_name, use_last_n_layers):
    if use_last_n_layers == -1:
        return
    current_count = 0
    stack = list(model.children())
    while stack:
        current_elem = stack.pop(0)
        children = list(current_elem.children())
        if len(children) == 0:
            current_count += 1
            if current_count < NUM_LAYERS[model_name] - use_last_n_layers + 1:
                for param in current_elem.parameters():
                    param.requires_grad = False
        else:
            stack = children + stack


def get_rgb_tiles_stats(rgb_tiles_paths: str = "data/rgb_tiles/*_jpg"):

    stats = {}

    for fp in glob.glob(rgb_tiles_paths):
        name = fp.split("/")[-1]
        print(name)
        channel_sum = np.zeros(3)
        count = 0
        for img_fp in glob.glob(path.join(fp, "*.jpg")):
            img = plt.imread(img_fp)
            channel_sum += (img / 255).sum(0).sum(0)
            count += img.shape[0] * img.shape[1]
        mean = (channel_sum / count).tolist()

        channel_sq_sum = np.zeros(3)
        for img_fp in glob.glob(path.join(fp, "*.jpg")):
            img = plt.imread(img_fp)
            channel_sq_sum += np.power(img / 255 - mean, 2).sum(0).sum(0)
            count += img.shape[0] * img.shape[1]
        std = (channel_sq_sum / count).tolist()
        stats[name] = {
            "mean": mean,
            "std": std
        }
    return stats


class RGBTilesDataset(Dataset):
    def __init__(self, root_dir: str, dataset_name: str, phase: str = "train",
                 use_transform: bool = True):
        assert phase in ["train", "val", "test"], "Phase not known."
        assert path.isdir(path.join(root_dir, dataset_name)), \
            "Dataset folder does not exist."
        assert path.isfile(path.join(root_dir, "stats.json")), \
            "stats.json in `root_dir` does not exist."

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.phase = phase
        self.input_size = INPUT_SIZE[dataset_name]
        self.use_transform = use_transform

        with open(path.join(root_dir, "stats.json")) as f:
            stats = json.load(f)[dataset_name]

        # Data augmentation and normalization for training
        # Just normalization for validation
        if self.use_transform:
            if phase == "train":
                self.transform_fn = transforms.Compose([
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(stats["mean"], stats["std"])
                ])
            else:
                self.transform_fn = transforms.Compose([
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(stats["mean"], stats["std"])
                ])
        self.data_tuples = []

        if phase == "train":
            # positives
            self.data_tuples += [
                (fp, 1) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*_train_pos_*.jpg"))]
            # negatives
            self.data_tuples += [
                (fp, 0) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*-neg-tr-*.jpg"))]
        elif phase == "val":
            # positives
            self.data_tuples += [
                (fp, 1) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*_val_pos_*.jpg"))]
            # negatives
            self.data_tuples += [
                (fp, 0) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*-neg-val-*.jpg"))]
        elif phase == "test":

            # positives
            self.data_tuples += [
                (fp, 1) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*_test_pos_*.jpg"))]
            # negatives
            self.data_tuples += [
                (fp, 0) for fp in glob.glob(
                    path.join(root_dir, dataset_name, "*-neg-te-*.jpg"))]
        print("Loaded paths for {} dataset ({} pos, {} neg)".format(
            phase,
            sum([1 for dt in self.data_tuples if dt[1] == 1]),
            sum([1 for dt in self.data_tuples if dt[1] == 0])
        ))

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        img_path, label = self.data_tuples[index]
        image = Image.open(img_path)
        if self.use_transform:
            image = self.transform_fn(image)
        return image, label


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Finetune hyperparams")
    argparser.add_argument("--output_dir", required=True, type=str)
    argparser.add_argument("--overwrite", action="store_true",
                           help="Overwrite existing output directory.")
    argparser.add_argument("--model_name", default="resnet50", type=str,
                           choices=["resnet50", "resnet18", "wide_resnet50_2",
                                    "efficientnet_v2_s", "efficientnet_v2_m"])
    argparser.add_argument("--num_classes", default=2, type=int)
    argparser.add_argument("--batch_size", default=64, type=int)
    argparser.add_argument("--num_epochs", default=100, type=int)
    argparser.add_argument("--use_last_n_layers", default=1, type=int)
    argparser.add_argument("--no_use_pretrained", action="store_true")
    argparser.add_argument("--num_workers", default=8, type=int)
    argparser.add_argument("--lr", default=1e-3, type=float)
    argparser.add_argument("--dataset_root_dir", type=str,
                           default="data/rgb_tiles")
    argparser.add_argument("--dataset_name",
                           default="v1_b2p_rgb_large_590_320_jpg", type=str,
                           choices=list(INPUT_SIZE.keys()))

    args = argparser.parse_args()

    if path.isdir(args.output_dir) and not args.overwrite:
        print("Path {} already exists.".format(args.output_dir))
        exit()
    elif args.overwrite:
        shutil.rmtree(args.output_dir)

    makedirs(args.output_dir)
    print("Output path: {}".format(args.output_dir))
    with open(path.join(args.output_dir, "opts.json"), "w+") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize the model for this run
    model = initialize_model(args.model_name, args.num_classes,
                             use_last_n_layers=args.use_last_n_layers,
                             use_pretrained=not args.no_use_pretrained)

    # Print the model we just instantiated
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable params: {}".format(params))

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    datasets = {
        x: RGBTilesDataset(args.dataset_root_dir, args.dataset_name, phase=x)
        for x in ['train', 'val', 'test']}
    # Create training and validation dataloaders
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    dataloaders["val"] = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    dataloaders["test"] = DataLoader(
        datasets["test"], batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(
        model, dataloaders, criterion, optimizer, lr_scheduler,
        path.join(args.output_dir, "best.cpkt"),
        path.join(args.output_dir, "last.cpkt"), num_epochs=args.num_epochs)

    stats = test_best_model(model, criterion, dataloaders)

    with open(path.join(args.output_dir, "test_stats.json"), "w+") as f:
        json.dump(stats, f, indent=4)

    pd.DataFrame(hist[1:], columns=hist[0]).to_csv(
        path.join(args.output_dir, "train_history.csv"), index=False)
