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


if __name__ == "__main__":
    parser = argparser.get_args(parse_args=False, add_save_dir=False)
    parser.add_argument("--save_dirs", required=True, type=str, nargs="+")
    args = parser.parse_args()

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

    dataloaders = get_dataloaders(
        args.batch_size, args.tile_size
    )
    _, dataloader_validation, dataloader_test = dataloaders
    data = {
        "val": {"imgs": [], "labs": []},
        "test": {"imgs": [], "labs": []},
    }
    for imgs, labs in dataloader_validation:
        data["val"]["imgs"].append(imgs)
        data["val"]["labs"].append(labs)
    for imgs, labs in dataloader_test:
        data["test"]["imgs"].append(imgs)
        data["test"]["labs"].append(labs)
    # val = [[batch_1, batch_2, batch_3], [batch_1, batch_2, batch_3]]
    pred_probs = {"val": [], "test": []}
    model_stats = {
        "val": {
            "acc": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0
        },
        "test": {
            "acc": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }
    }
    num_models = 0
    for save_dir in args.save_dirs:
        if not path.isdir(save_dir):
            print("Save dir {} does not exist.".format(save_dir))
            continue
        model_fp = path.join(save_dir, "best.pt")
        if not path.isfile(model_fp):
            print("Model state dict {} does not exist.".format(
                path.join(save_dir, "best.pt")))
            continue
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['model_state_dict'])
        if cuda:
            model = model.cuda()
        model.eval()
        print(
            "Loaded model from {} at epoch {}".format(
                model_fp, checkpoint["epoch"]) +
            " with val loss {:.4} and acc {:.2f}".format(
                checkpoint["val_loss"], checkpoint["val_acc"]))
        with torch.no_grad():
            for name, data_dict in data.items():
                pred_probs[name].append([])
                num_samples = 0
                for imgs, labs in zip(data_dict["imgs"], data_dict["labs"]):
                    if cuda:
                        imgs = imgs.float().cuda()
                        labs = labs.float().cuda()
                    logits = model(imgs)
                    num_samples += imgs.shape[0]
                    preds = torch.argmax(logits, -1)
                    pred_probs[name][-1].append(
                        torch.softmax(logits, -1).cpu())
                    model_stats[name]["acc"] += (preds == labs).sum().item()
                    model_stats[name]["tp"] += torch.logical_and(
                        preds == 1, labs == 1).sum()
                    model_stats[name]["fp"] += torch.logical_and(
                        preds == 1, labs == 0).sum()
                    model_stats[name]["tn"] += torch.logical_and(
                        preds == 0, labs == 0).sum()
                    model_stats[name]["fn"] += torch.logical_and(
                        preds == 0, labs == 1).sum()
                model_stats[name]["acc"] /= num_samples
            print("Val acc: {:.2f}% test acc: {:.2f}%".format(
                model_stats["val"]["acc"] * 100,
                model_stats["test"]["acc"] * 100))
            print("Val confusion matrix\t\tTest confusion matrix")
            print("{:04d} (TP) {:04d} (FP)\t\t{:04d} (TP) {:04d} (FP)".format(
                model_stats["val"]["tp"], model_stats["val"]["fp"],
                model_stats["test"]["tp"], model_stats["test"]["fp"]))
            print("{:04d} (FN) {:04d} (TN)\t\t{:04d} (TP) {:04d} (FP)".format(
                model_stats["val"]["fn"], model_stats["val"]["tn"],
                model_stats["test"]["fn"], model_stats["test"]["tn"]))
        num_models += 1

    for name in ["val", "test"]:
        num_batch = len(pred_probs[name][0])
        acc, tp, fp, tn, fn, num_samples = 0, 0, 0, 0, 0, 0
        for i in range(num_batch):
            preds = torch.mean(
                torch.cat([
                    pred_probs[name][j][i].unsqueeze(1)
                    for j in range(num_models)], 1),
                1)
            num_samples += preds.shape[0]
            preds = torch.argmax(preds, -1)
            labs = data[name]["labs"][i]
            acc += (preds == labs).sum().item()

            tp += torch.logical_and(preds == 1, labs == 1).sum()
            fp += torch.logical_and(preds == 1, labs == 0).sum()
            tn += torch.logical_and(preds == 0, labs == 0).sum()
            fn += torch.logical_and(preds == 0, labs == 1).sum()
        print("Ensemble {} acc: {:.2f}".format(name, acc / num_samples * 100))
        print("{} confusion matrix".format(name))
        print("{:04d} (TP) {:04d} (FP)".format(tp, fp))
        print("{:04d} (FN) {:04d} (TN)".format(fn, tn))
