import argparse
import json
import numpy as np
import os
import time

import os.path as path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from src import models, utils
from src.data.bridge_site import get_num_channels, get_dataloaders

from test_ssl import do_test


def test(model: nn.Module, criterion: nn.modules.loss._Loss,
         dataloader, cuda: bool, no_use_several_test_samples: bool = False,
         name: str = "train"):
    start_time = time.time()
    model.eval()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()

    tp_tn_fp_fn = np.zeros(4)
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
            tp_tn_fp_fn += np.array(utils.get_tp_tn_fp_fn(
                prediction.detach().cpu(), labels.cpu()))

            losses.update(loss.item(), inputs.size(0))
            prec1 = utils.accuracy(prediction, labels, topk=[1])[0]
            accs.update(prec1.item(), inputs.size(0))
    time_taken = time.time() - start_time
    tp_tn_fp_fn = list(map(int, tp_tn_fp_fn.tolist()))
    f1 = utils.get_f1(*tp_tn_fp_fn)
    print(
        "[{}] loss: {:.4f} acc: {:.4f} F1: {:.4f}".format(
            name, losses.avg, accs.avg, f1) +
        " TP: {:d} TN: {:d} FP: {:d} FN: {:d} ({:.2f} sec.)".format(
            *tp_tn_fp_fn, time_taken))
    return {
        "loss": float(np.round(losses.avg, 8)),
        "acc": float(np.round(accs.avg, 8)),
        "f1": float(np.round(f1, 8)),
        "tp": tp_tn_fp_fn[0],
        "tn": tp_tn_fp_fn[1],
        "fp": tp_tn_fp_fn[2],
        "fn": tp_tn_fp_fn[3],
    }


def main(args):
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    cuda = not args.no_cuda and torch.cuda.is_available()

    print("git:\n  {}\n".format(utils.get_sha()))
    utils.print_args(args)

    # model
    num_channels = get_num_channels(args.data_modalities)
    model = models.BridgeResnet(
        model_name=args.model, lazy=False, num_channels=num_channels)
    best_model_fp = path.join(args.save_dir, "model_best.pt")
    checkpoint = torch.load(best_model_fp, map_location="cpu")
    model_checkpoint = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            model_checkpoint[k.replace("module.", "")] = v
        else:
            model_checkpoint[k] = v
    model.load_state_dict(model_checkpoint)
    print(
        "Loaded model at epoch {} with ".format(checkpoint["epoch"]) +
        "val loss {:.4f} and val acc {:.4f}.".format(
            checkpoint["val_loss"], checkpoint["val_acc"]))

    # loss
    criterion = nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataloaders = get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=not args.no_use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version,
        data_order=args.data_modalities,
        ddp=False
    )
    (dataloader_train, dataloader_validation, dataloader_test_rw,
     dataloader_test_ug, _) = dataloaders

    tr_stats = test(
        model, criterion, dataloader_train, cuda, name="Train",
        no_use_several_test_samples=True)
    va_stats = test(
        model, criterion, dataloader_validation, cuda, name="Val",
        no_use_several_test_samples=args.no_use_several_test_samples)
    te_rw_stats = test(
        model, criterion, dataloader_test_rw, cuda, name="Test (Rwanda)",
        no_use_several_test_samples=args.no_use_several_test_samples)
    te_ug_stats = test(
        model, criterion, dataloader_test_ug, cuda, name="Test (Uganda)",
        no_use_several_test_samples=args.no_use_several_test_samples)

    stats = {"tr_" + k: v for k, v in tr_stats.items()}
    stats.update({"va_" + k: v for k, v in va_stats.items()})
    stats.update({"te_rw_" + k: v for k, v in te_rw_stats.items()})
    stats.update({"te_ug_" + k: v for k, v in te_ug_stats.items()})
    with open(path.join(args.save_dir, "test_stats.json"), "w+") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument for training.")
    # batch size
    parser.add_argument("--test_batch_size", default=8, type=int)
    # save directory
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if do_test(args.save_dir, overwrite=args.overwrite):
        with open(path.join(args.save_dir, "opts.json")) as f:
            opts = json.load(f)
        opts["no_cuda"] = args.no_cuda
        opts["test_batch_size"] = args.test_batch_size
        opts = utils.Args(**opts)
        main(opts)
