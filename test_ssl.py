"""
Extract train, val, test statistics from trained SSL models.
"""
import argparse
import json
import time

import os.path as path

import torch

import torch.nn as nn

import torch.backends.cudnn as cudnn

from src import models
from train import get_num_channels
from train_ssl import VAL_LOG_FORMAT

from src.data import get_dataloaders

from third_party.MixMatch.utils import AverageMeter, accuracy


def do_test(save_dir, overwrite=False):
    """Checks whether model file paths exists and the training has finished.
    """
    if not path.isdir(save_dir):
        print("{} does not exist.".format(save_dir))
        return False

    opts_fp = path.join(save_dir, "opts.json")
    if not path.isfile(opts_fp):
        print("{} does not exist.".format(opts_fp))
        return False

    with open(opts_fp, "r") as f:
        opts = json.load(f)
    if not opts["finished"]:
        print("According to opts.json the model has not finished.")
        return False

    best_model_fp = path.join(save_dir, "model_best.pth.tar")
    if not path.isfile(best_model_fp):
        print("{} does not exist.".format(best_model_fp))
        return False
    test_stats_fp = path.join(save_dir, "test_stats.json")
    if path.isfile(test_stats_fp) and not overwrite:
        print("{} does not exist.".format(test_stats_fp))
        return False

    return True


def test(dataloader, model, criterion, mode, cuda: bool = True,
         use_several_test_samples: bool = False):
    start = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    tp, tn, fp, fn = 0, 0, 0, 0

    def log():
        print(VAL_LOG_FORMAT.format(
            total=time.time() - start,
            name=mode,
            batch=batch_idx + 1,
            size=len(dataloader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            top1=top1.avg,
        ))

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.float()
            batch_size = inputs.shape[0]
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            if use_several_test_samples:
                batch_size, num_samples, c, w, h = inputs.shape
                inputs_ = inputs.view(batch_size * num_samples, c, w, h)
                targets_ = targets.view(-1, 1).repeat(1, num_samples).view(
                    batch_size * num_samples)
            else:
                inputs_ = inputs
                targets_ = targets
            # compute output
            outputs = model(inputs_)
            loss = criterion(outputs, targets_)
            # measure accuracy and record loss
            if use_several_test_samples:
                outputs_ = torch.softmax(
                    outputs.view(batch_size, num_samples, -1), -1).mean(1)
                loss = loss.view(batch_size, num_samples).mean(1).mean()
            else:
                outputs_ = outputs
                loss = loss.mean()

            prec1 = accuracy(outputs_, targets, topk=[1])[0]
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            preds = torch.argmax(outputs_, 1)
            tp += torch.logical_and(
                targets == 1, preds == targets).sum().item()
            tn += torch.logical_and(
                targets == 0, preds == targets).sum().item()
            fp += torch.logical_and(
                targets == 1, preds != targets).sum().item()
            fn += torch.logical_and(
                targets == 0, preds != targets).sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    log()
    return (losses.avg, top1.avg, tp, tn, fp, fn)


def main(args):
    # cuda
    cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    # load hyperparameters
    opts_fp = path.join(args.save_dir, "opts.json")
    with open(opts_fp, "r") as f:
        model_args = json.load(f)

    # get the model
    num_channels = get_num_channels(model_args["data_modalities"])
    ema_model = models.BridgeResnet(
        model_name=model_args["model"], lazy=False,
        num_channels=num_channels)

    # load model with pretrained weights
    best_model_fp = path.join(args.save_dir, "model_best.pth.tar")
    checkpoint = torch.load(best_model_fp, map_location="cpu")
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    print(
        "Loaded model from epoch {epoch} ".format(epoch=checkpoint["epoch"]) +
        "with validation accuracy {acc:.2f}".format(
            acc=checkpoint["best_acc"]))
    # criterion
    criterion = nn.CrossEntropyLoss(reduction="none")
    # move to gpu
    if cuda:
        ema_model = ema_model.cuda()
        criterion = criterion.cuda()

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.batch_size, model_args["tile_size"],
        use_augment=not model_args["no_augmentation"],
        use_several_test_samples=model_args["use_several_test_samples"],
        num_test_samples=model_args["num_test_samples"],
        test_batch_size=args.test_batch_size,
        data_version=model_args["data_version"],
        data_order=model_args["data_modalities"]
    )

    tr_avg_loss, tr_avg_top1, tr_tp, tr_tn, tr_fp, tr_fn = test(
        train_loader, ema_model, criterion, mode="Train", cuda=cuda,
        use_several_test_samples=False)
    va_avg_loss, va_avg_top1, va_tp, va_tn, va_fp, va_fn = test(
        val_loader, ema_model, criterion, mode="Validation", cuda=cuda,
        use_several_test_samples=model_args["use_several_test_samples"])
    te_avg_loss, te_avg_top1, te_tp, te_tn, te_fp, te_fn = test(
        test_loader, ema_model, criterion, mode="Test", cuda=cuda,
        use_several_test_samples=model_args["use_several_test_samples"])

    stats = {
        "train": {
            "loss": tr_avg_loss,
            "acc": tr_avg_top1,
            "tp": tr_tp,
            "tn": tr_tn,
            "fp": tr_fp,
            "fn": tr_fn,
        },
        "val": {
            "loss": va_avg_loss,
            "acc": va_avg_top1,
            "tp": va_tp,
            "tn": va_tn,
            "fp": va_fp,
            "fn": va_fn,
        },
        "test": {
            "loss": te_avg_loss,
            "acc": te_avg_top1,
            "tp": te_tp,
            "tn": te_tn,
            "fp": te_fp,
            "fn": te_fn,
        },
    }

    test_stats_fp = path.join(args.save_dir, "test_stats.json")
    with open(test_stats_fp, "w+") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--test_batch_size", default=10, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if do_test(args.save_dir, overwrite=args.overwrite):
        main(args)
