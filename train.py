import json
import os
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src import models
from src.utils import argparser, dist
from src.data import get_dataloaders


def train_for_an_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: DataLoader,
    optimizer,
    cuda: bool = True,
    log_interval: int = None,
):
    model.train()
    epoch_loss = 0
    num_train = 0
    acc = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

        prediction = model(inputs)
        loss = criterion(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc += (torch.argmax(prediction, -1) == labels).sum()
        epoch_loss += loss.item() * inputs.shape[0]
        num_train += inputs.shape[0]
        tp += torch.logical_and(torch.argmax(prediction, -1) == 1, labels == 1).sum()
        fp += torch.logical_and(torch.argmax(prediction, -1) == 1, labels == 0).sum()
        tn += torch.logical_and(torch.argmax(prediction, -1) == 0, labels == 0).sum()
        fn += torch.logical_and(torch.argmax(prediction, -1) == 0, labels == 1).sum()
        if log_interval is not None and i % log_interval == 0:
            print(f"Step loss: {loss.item()}")
    return epoch_loss / num_train, acc / num_train, (tp, fp, tn, fn)


def test_for_an_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader,
    cuda: bool,
):
    model.eval()
    epoch_loss = 0
    num_test = 0
    acc_test = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

        prediction = model(inputs)
        loss = criterion(prediction, labels)
        acc_test += (torch.argmax(prediction, -1) == labels).sum()
        tp += torch.logical_and(torch.argmax(prediction, -1) == 1, labels == 1).sum()
        fp += torch.logical_and(torch.argmax(prediction, -1) == 1, labels == 0).sum()
        tn += torch.logical_and(torch.argmax(prediction, -1) == 0, labels == 0).sum()
        fn += torch.logical_and(torch.argmax(prediction, -1) == 0, labels == 1).sum()

        epoch_loss += loss.item() * inputs.shape[0]
        num_test += inputs.shape[0]
    return epoch_loss / num_test, acc_test / num_test, (tp, fp, tn, fn)


def run(
    model: nn.Module,
    criterion: nn.Module,
    dataloaders,
    optimizer,
    epochs: int,
    save_dir: str,
    log_interval: int = None,
    cuda: bool = True,
):
    dataloader_train, dataloader_validation, dataloader_test = dataloaders
    best_val_loss, best_epoch = float("inf"), -1
    best_save_fp = path.join(save_dir, "best.pt")
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch}")
        epoch_loss_train, acc_train, stats_train = train_for_an_epoch(
            model,
            criterion,
            dataloader_train,
            optimizer,
            log_interval=log_interval,
            cuda=cuda,
        )
        with torch.no_grad():
            epoch_loss_validation, acc_val, stats_val = test_for_an_epoch(
                model, criterion, dataloader_validation, epoch, cuda
            )
            epoch_loss_test, acc_test, stats_test = test_for_an_epoch(
                model, criterion, dataloader_test, epoch, cuda
            )
        print(
            "[Train epoch {:d}] loss: {:.4f} acc: {:.2f}%".format(
                epoch, epoch_loss_train, acc_train * 100
            )
        )
        print(
            "[Val epoch {:d}] loss: {:.4f} acc: {:.2f}%".format(
                epoch, epoch_loss_validation, acc_val * 100
            )
            + " best loss: {:.4f} (epoch {})".format(best_val_loss, best_epoch)
        )
        print(
            "[Test epoch {:d}] loss: {:.4f} acc: {:.2f}%".format(
                epoch, epoch_loss_test, acc_test * 100
            )
        )
        print("Train confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_train[0], stats_train[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_train[3], stats_train[2]))
        print("Val confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_val[0], stats_val[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_val[3], stats_val[2]))
        print("Test confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_test[0], stats_test[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_test[3], stats_test[2]))
        if epoch_loss_validation < best_val_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": epoch_loss_validation,
                    "val_acc": acc_val,
                },
                best_save_fp,
            )
            best_val_loss = epoch_loss_validation
            best_epoch = epoch
            print("Save to {}".format(best_save_fp))


def train(args):
    # init distributed mode
    cuda = torch.cuda.is_available()
    if cuda:
        dist.init_distributed_mode(args)
    dist.fix_random_seeds(args.seed, cuda=cuda)

    cudnn.benchmark = True
    # model
    model = models.BridgeResnet(model_name=args.model)
    # loss
    criterion = nn.CrossEntropyLoss()
    # ddp, cuda
    if dist.has_batchnorms(model) and cuda:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataloaders = get_dataloaders(args.batch_size, args.tile_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run(
        model,
        criterion,
        dataloaders,
        optimizer,
        args.epochs,
        args.save_dir,
        log_interval=args.log_interval,
        cuda=cuda,
    )


if __name__ == "__main__":
    args = argparser.get_args()
    print("git:\n  {}\n".format(dist.get_sha()))
    print("Args:")
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    if path.isdir(args.save_dir):
        print("Save dir {} already exists.".format(args.save_dir))
    else:
        os.makedirs(args.save_dir)
        with open(path.join(args.save_dir, "opts.json"), "w+") as f:
            json.dump(vars(args), f, indent=4)
        train(args)
