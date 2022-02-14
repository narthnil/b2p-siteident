import argparse
import json
import os
import time
import os.path as path
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src import models, utils
from src.data import bridge_site
from src.utils import AverageMeter, accuracy
from src.data.bridge_site import get_num_channels


def train_for_an_epoch(model: nn.Module, criterion: nn.modules.loss._Loss,
                       dataloader: DataLoader, optimizer, cuda: bool = True,
                       log_interval: int = None):
    start_time = time.time()
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    tp_tn_fp_fn = np.zeros(4)
    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

        prediction = model(inputs)
        loss = criterion(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        prec1 = accuracy(prediction, labels, topk=[1])[0]

        accs.update(prec1.item(), inputs.size(0))
        tp_tn_fp_fn += np.array(utils.get_tp_tn_fp_fn(
            prediction.detach().cpu(), labels.cpu()))

        if log_interval is not None and i % log_interval == 0:
            print(f"Step loss: {loss.item()}")
    time_taken = time.time() - start_time

    return (
        float(np.round(losses.avg, 8)),
        float(np.round(accs.avg, 8)),
        list(map(int, tp_tn_fp_fn.tolist())),
        float(np.round(time_taken, 2)))


def test_for_an_epoch(model: nn.Module, criterion: nn.modules.loss._Loss,
                      dataloader, cuda: bool,
                      no_use_several_test_samples: bool = False):
    start_time = time.time()
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    tp_fp_tn_fn = np.zeros(4)
    with torch.no_grad():
        for inputs, labels in dataloader:
            if cuda:
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
            if not no_use_several_test_samples:
                batch_size, num_samples, c, w, h = inputs.shape
                inputs = inputs.view(batch_size * num_samples, c, w, h)

            prediction = model(inputs)

            if not no_use_several_test_samples:
                labels_ = labels.unsqueeze(-1).repeat(1, num_samples).view(-1)
                loss = criterion(prediction, labels_)
            else:
                loss = criterion(prediction, labels)
            if not no_use_several_test_samples:
                prediction = torch.softmax(
                    prediction.view(batch_size, num_samples, -1), -1).mean(1)
            tp_fp_tn_fn += np.array(utils.get_tp_tn_fp_fn(
                prediction.detach().cpu(), labels.cpu()))

            losses.update(loss.item(), inputs.size(0))
            prec1 = accuracy(prediction, labels, topk=[1])[0]
            accs.update(prec1.item(), inputs.size(0))
    time_taken = time.time() - start_time
    return (
        float(np.round(losses.avg, 8)),
        float(np.round(accs.avg, 8)),
        list(map(int, tp_fp_tn_fn.tolist())),
        float(np.round(time_taken, 2)))


def train(model: nn.Module, criterion: nn.Module, dataloaders: Tuple,
          optimizer, epochs: int, save_dir: str, cuda: bool = True,
          log_interval: int = None, scheduler=None,
          no_use_several_test_samples: bool = False):
    (dataloader_train, dataloader_validation, dataloader_test_rw,
     dataloader_test_ug, _) = dataloaders
    best_val_loss, best_epoch = float("inf"), -1
    best_save_fp = path.join(save_dir, "best.pt")
    train_logs = [("epoch", "train_loss", "train_acc", "train_time",
                   "val_loss", "val_acc", "val_time", "test_rw_loss",
                   "test_rw_acc", "test_rw_time", "test_ug_loss",
                   "test_ug_acc", "test_ug_time", "best_val_loss",
                   "best_val_epoch", "tr_tp", "tr_tn", "tr_fp", "tr_fn",
                   "va_tp", "va_tn", "va_fp", "va_fn", "te_rw_tp", "te_rw_tn",
                   "te_rw_fp", "te_rw_fn", "te_ug_tp", "te_ug_tn", "te_ug_fp",
                   "te_ug_fn")]
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch}")
        loss_train, acc_train, stats_train, time_train = train_for_an_epoch(
            model, criterion, dataloader_train, optimizer,
            log_interval=log_interval, cuda=cuda,
        )
        loss_val, acc_val, stats_val, time_val = test_for_an_epoch(
            model, criterion, dataloader_validation, cuda,
            no_use_several_test_samples=no_use_several_test_samples
        )
        (loss_test_rw, acc_test_rw, stats_test_rw,
         time_test_rw) = test_for_an_epoch(
            model, criterion, dataloader_test_rw, cuda,
            no_use_several_test_samples=no_use_several_test_samples
        )
        (loss_test_ug, acc_test_ug, stats_test_ug,
         time_test_ug) = test_for_an_epoch(
            model, criterion, dataloader_test_ug, cuda,
            no_use_several_test_samples=no_use_several_test_samples
        )
        print(("[Train epoch {:d}] loss: {:.4f} acc: {:.2f}% F1: {:.4f}"
               "time: {:.1f} sec.").format(
            epoch, loss_train, acc_train, utils.get_f1(*stats_train),
            time_train))
        print(("[Val epoch {:d}] loss: {:.4f} acc: {:.2f}% F1: {:.4f} "
               "time: {:.1f} sec. best loss: {:.4f} (epoch {})").format(
            epoch, loss_val, acc_val, utils.get_f1(*stats_val), time_val,
            best_val_loss, best_epoch))
        print(("[Test (Rwanda) epoch {:d}] loss: {:.4f} acc: {:.2f}% "
               "F1: {:.4f} sec. time: {:.1f} sec.").format(
            epoch, loss_test_rw, acc_test_rw, utils.get_f1(*stats_test_rw),
            time_test_rw))
        print(("[Test (Uganda) epoch {:d}] loss: {:.4f} acc: {:.2f}% "
               "F1: {:.4f} sec. time: {:.1f} sec.").format(
            epoch, loss_test_ug, acc_test_ug, utils.get_f1(*stats_test_ug),
            time_test_ug))

        # utils.print_confusion_matrix(*stats_train, name="Train")
        # utils.print_confusion_matrix(*stats_val, name="Val")
        # utils.print_confusion_matrix(*stats_test_rw, name="Test Rwanda")
        # utils.print_confusion_matrix(*stats_test_ug, name="Test Uganda")

        train_log = [epoch, loss_train, acc_train, time_train, loss_val,
                     acc_val, time_val, loss_test_rw, acc_test_rw,
                     time_test_rw, loss_test_ug, acc_test_ug, time_test_ug,
                     best_val_loss, best_epoch]
        train_log += list(stats_train) + list(stats_val) + list(
            stats_test_rw) + list(stats_test_ug)
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
        else:
            print("Model can be found under", args.save_dir)
        if scheduler is not None:
            scheduler.step(loss_val)

    return train_logs


def main(args):
    # init distributed mode
    cuda = torch.cuda.is_available()
    utils.fix_random_seeds(args.seed)

    cudnn.benchmark = True
    # model
    num_channels = get_num_channels(args.data_modalities)
    model = models.BridgeResnet(
        model_name=args.model, lazy=False, num_channels=num_channels)
    # loss
    criterion = nn.CrossEntropyLoss()
    # cuda
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataloaders = bridge_site.get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=not args.no_use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version,
        data_order=args.data_modalities
    )

    if args.use_sgd_scheduler:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    train_logs = train(
        model,
        criterion,
        dataloaders,
        optimizer,
        args.epochs,
        args.save_dir,
        cuda=cuda,
        log_interval=args.log_interval,
        scheduler=scheduler,
        no_use_several_test_samples=args.no_use_several_test_samples
    )
    return train_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # model
    parser.add_argument("--model", type=str,
                        choices=["resnet18", "resnet50", "resnext",
                                 "efficientnet_b2", "efficientnet_b7"],
                        default="efficientnet_b2")

    # data
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of samples per training step")
    parser.add_argument("--tile_size", type=int,
                        choices=[300, 600, 1200], default=300)
    parser.add_argument("--data_version", default="v1", type=str,
                        choices=["v1", "v2"])
    parser.add_argument("--data_modalities", nargs="+", type=str,
                        default=["population", "osm_img", "elevation",
                                 "slope", "roads", "waterways",
                                 "admin_bounds_qgis"])
    parser.add_argument("--no_augmentation", action="store_true")

    parser.add_argument("--no_use_several_test_samples", action="store_true")
    parser.add_argument("--num_test_samples", default=32, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)

    # log during training
    parser.add_argument("--log_interval", default=None, type=int,
                        help=("Whether to log every `log_interval`-th "
                              "iterations."))

    # training
    parser.add_argument("--epochs", default=200, type=int,
                        help="Training epochs.")

    # optimizer
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Adam optimizer learning rate.")
    parser.add_argument("--use_sgd_scheduler", action="store_true")
    # save directory
    parser.add_argument("--save_dir", required=True, type=str)

    args = parser.parse_args()

    utils.print_args(args)

    if path.isdir(args.save_dir):
        print("Save dir {} already exists.".format(args.save_dir))
    else:
        os.makedirs(args.save_dir)
        with open(path.join(args.save_dir, "opts.json"), "w+") as f:
            json.dump(vars(args), f, indent=4)
        train_logs = main(args)
        with open(path.join(args.save_dir, "train_logs.csv"), "w+") as f:
            f.write("\n".join([",".join(map(str, l)) for l in train_logs]))
