import json
import os
import time
import os.path as path
import numpy as np

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
    start_time = time.time()
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
        tp += torch.logical_and(torch.argmax(prediction, -1)
                                == 1, labels == 1).sum()
        fp += torch.logical_and(torch.argmax(prediction, -1)
                                == 1, labels == 0).sum()
        tn += torch.logical_and(torch.argmax(prediction, -1)
                                == 0, labels == 0).sum()
        fn += torch.logical_and(torch.argmax(prediction, -1)
                                == 0, labels == 1).sum()

        if log_interval is not None and i % log_interval == 0:
            print(f"Step loss: {loss.item()}")
    time_taken = time.time() - start_time
    return (
        float(np.round(epoch_loss / num_train, 8)),
        float(np.round(acc.item() / num_train, 8)),
        (tp.item(), fp.item(), tn.item(), fn.item()),
        float(np.round(time_taken, 2)))


def test_for_an_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader,
    cuda: bool,
    use_several_test_samples: bool = False
):
    start_time = time.time()
    model.eval()
    epoch_loss = 0
    num_test = 0
    acc = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for inputs, labels in dataloader:
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
        if use_several_test_samples:
            batch_size, num_samples, c, w, h = inputs.shape
            inputs = inputs.view(batch_size * num_samples, c, w, h)

        prediction = model(inputs)

        if use_several_test_samples:
            labels_ = labels.unsqueeze(-1).repeat(1, num_samples).view(-1)
            loss = criterion(prediction, labels_)
        else:
            loss = criterion(prediction, labels)
        if use_several_test_samples:
            prediction = torch.softmax(
                prediction.view(batch_size, num_samples, -1), -1).mean(1)
        acc += (torch.argmax(prediction, -1) == labels).sum()
        tp += torch.logical_and(torch.argmax(prediction, -1)
                                == 1, labels == 1).sum()
        fp += torch.logical_and(torch.argmax(prediction, -1)
                                == 1, labels == 0).sum()
        tn += torch.logical_and(torch.argmax(prediction, -1)
                                == 0, labels == 0).sum()
        fn += torch.logical_and(torch.argmax(prediction, -1)
                                == 0, labels == 1).sum()

        epoch_loss += loss.item() * labels.shape[0]
        num_test += labels.shape[0]
    time_taken = time.time() - start_time
    return (
        float(np.round(epoch_loss / num_test, 8)),
        float(np.round(acc.item() / num_test, 8)),
        (tp.item(), fp.item(), tn.item(), fn.item()),
        float(np.round(time_taken, 2)))


def train(
    model: nn.Module,
    criterion: nn.Module,
    dataloaders,
    optimizer,
    epochs: int,
    save_dir: str,
    log_interval: int = None,
    cuda: bool = True,
    use_several_test_samples: bool = False
):
    dataloader_train, dataloader_validation, dataloader_test, _ = dataloaders
    best_val_loss, best_epoch = float("inf"), -1
    best_save_fp = path.join(save_dir, "best.pt")
    train_logs = [("epoch", "train_loss", "train_acc", "train_time",
                   "val_loss", "val_acc", "val_time", "test_loss", "test_acc",
                  "test_time", "tr_tp", "tr_fp", "tr_tn", "tr_fp", "va_tp",
                   "va_fp", "va_tn", "va_fp", "te_tp", "te_fp", "te_tn",
                   "te_fp")]
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch}")
        loss_train, acc_train, stats_train, time_train = train_for_an_epoch(
            model, criterion, dataloader_train, optimizer,
            log_interval=log_interval, cuda=cuda,
        )
        with torch.no_grad():
            loss_val, acc_val, stats_val, time_val = test_for_an_epoch(
                model, criterion, dataloader_validation, cuda,
                use_several_test_samples=use_several_test_samples
            )
            loss_test, acc_test, stats_test, time_test = test_for_an_epoch(
                model, criterion, dataloader_test, cuda,
                use_several_test_samples=use_several_test_samples
            )
        print(("[Train epoch {:d}] loss: {:.4f} acc: {:.2f}% "
               "time: {:.1f} sec.").format(
            epoch, loss_train, acc_train * 100, time_train))
        print(("[Val epoch {:d}] loss: {:.4f} acc: {:.2f}% time: {:.1f} sec."
               "best loss: {:.4f} (epoch {})").format(
            epoch, loss_val, acc_val * 100, time_val, best_val_loss,
            best_epoch))
        print(("[Test epoch {:d}] loss: {:.4f} acc: {:.2f}% "
               "time: {:.1f} sec.").format(
            epoch, loss_test, acc_test * 100, time_test))
        print("Train confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_train[0], stats_train[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_train[3], stats_train[2]))
        print("Val confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_val[0], stats_val[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_val[3], stats_val[2]))
        print("Test confusion matrix")
        print("{:04d} (TP) {:04d} (FP)".format(stats_test[0], stats_test[1]))
        print("{:04d} (FN) {:04d} (TN)".format(stats_test[3], stats_test[2]))

        train_log = [epoch, loss_train, acc_train, time_train, loss_val,
                     acc_val, time_val, loss_test, acc_test,
                     time_test]
        train_log += list(stats_train) + list(stats_val) + list(stats_test)
        train_logs.append(train_log)
        if loss_val < best_val_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": loss_val,
                    "val_acc": acc_val,
                },
                best_save_fp,
            )
            best_val_loss = loss_val
            best_epoch = epoch
            print("Save to {}".format(best_save_fp))

    return train_logs


def main(args):
    # init distributed mode
    cuda = torch.cuda.is_available()
    if cuda:
        dist.init_distributed_mode(args)
    dist.fix_random_seeds(args.seed, cuda=cuda)

    cudnn.benchmark = True
    # model
    model = models.BridgeResnet(model_name=args.model, lazy=False)
    # loss
    criterion = nn.CrossEntropyLoss()
    # ddp, cuda
    if dist.has_batchnorms(model) and cuda:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataloaders = get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=args.use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_logs = train(
        model,
        criterion,
        dataloaders,
        optimizer,
        args.epochs,
        args.save_dir,
        log_interval=args.log_interval,
        cuda=cuda,
        use_several_test_samples=args.use_several_test_samples
    )
    return train_logs


if __name__ == "__main__":
    args = argparser.get_args()
    print("git:\n  {}\n".format(dist.get_sha()))
    print("Args:")
    print(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )
    if path.isdir(args.save_dir):
        print("Save dir {} already exists.".format(args.save_dir))
    else:
        os.makedirs(args.save_dir)
        with open(path.join(args.save_dir, "opts.json"), "w+") as f:
            json.dump(vars(args), f, indent=4)
        train_logs = main(args)
        with open(path.join(args.save_dir, "train_logs.csv"), "w+") as f:
            f.write("\n".join([",".join(map(str, l)) for l in train_logs]))
