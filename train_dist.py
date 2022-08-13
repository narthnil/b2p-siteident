import argparse
from genericpath import isfile
import json
import os
import os.path as path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import models

from train_baseline import set_parameter_requires_grad, Args
from src import utils
from src.data.bridge_site import get_num_channels, get_dataloaders

from sklearn.metrics import balanced_accuracy_score, f1_score


def initialize_model(model_name, num_classes, num_channels, 
                     use_last_n_layers=-1, use_pretrained=True):
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
        model.features[0][0] = nn.Conv2d(
                num_channels, model.features[0][0].out_channels, 
                kernel_size=model.features[0][0].kernel_size, 
                stride=model.features[0][0].stride,
                padding=model.features[0][0].padding, 
                bias=model.features[0][0].bias)
        for param in model.features.parameters():
            param.requires_grad = True
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes))
    else:
        model.conv1 = nn.Conv2d(
            num_channels, model.conv1.out_channels, 
            kernel_size=model.conv1.kernel_size, 
            stride=model.conv1.stride)
        for param in model.conv1.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_for_an_epoch(model: nn.Module, criterion: nn.modules.loss._Loss,
                       dataloader: DataLoader, optimizer, epoch: int,
                       epochs: int, log_interval: int = None,
                       fp16_scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Train | Epoch: [{}/{}]'.format(epoch + 1, epochs)
    for (inputs, labels) in metric_logger.log_every(
            dataloader, log_interval, header):
        inputs = inputs.float().cuda()
        labels = labels.long().cuda()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            prediction = model(inputs)
            loss = criterion(prediction, labels)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        torch.cuda.synchronize()

        prec1 = utils.accuracy(prediction, labels, topk=[1])[0]
        n = inputs.shape[0]
        metric_logger.update(loss=(loss.item(), n), prec1=(prec1, n))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('[Train] | Epoch: [{}/{}]'.format(epoch + 1, epochs), metric_logger)
    global_avg = {k: (meter.global_avg, meter.count) for k,
                  meter in metric_logger.meters.items()}
    return global_avg


def test_for_an_epoch(model: nn.Module, criterion: nn.modules.loss._Loss,
                      dataloader: DataLoader, epoch: int, epochs: int,
                      no_use_several_test_samples: bool = False,
                      name: str = "Val"):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '{} | Epoch: [{}/{}]'.format(name, epoch + 1, epochs)
    with torch.no_grad():
        for inputs, labels in metric_logger.log_every(
                dataloader, len(dataloader) + 5, header):
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            if not no_use_several_test_samples:
                batch_size, num_samples, c, w, h = inputs.shape
                inputs = inputs.view(batch_size * num_samples, c, w, h)
            else:
                batch_size = inputs.shape[0]

            prediction = model(inputs)

            if not no_use_several_test_samples:
                labels_ = labels.unsqueeze(-1).repeat(1, num_samples).view(-1)
                loss = criterion(prediction, labels_)
            else:
                loss = criterion(prediction, labels)
            if not no_use_several_test_samples:
                prediction = torch.softmax(
                    prediction.view(batch_size, num_samples, -1), -1).mean(1)
            prec1 = utils.accuracy(prediction, labels, topk=[1])[0]
            metric_logger.update(
                loss=(loss.item(), batch_size), prec1=(prec1, batch_size))
    metric_logger.synchronize_between_processes()
    print('{} | Epoch: [{}/{}]'.format(name, epoch + 1, epochs), metric_logger)
    global_avg = {k: (meter.global_avg, meter.count) for k,
                  meter in metric_logger.meters.items()}
    return global_avg


def train(model: nn.Module, criterion: nn.Module, dataloaders: Tuple,
          optimizer, epochs: int, save_dir: str, log_interval: int = None,
          scheduler=None, no_use_several_test_samples: bool = False,
          no_test_set_eval: bool = False, fp16_scaler=None):

    best_val_loss, best_epoch = float("inf"), -1
    best_save_fp = path.join(save_dir, "model_best.pt")

    (dataloader_train, dataloader_validation, dataloader_test,
     dataloader_test_rw, dataloader_test_ug, _) = dataloaders
    stats_dict = []
    for epoch in range(epochs):
        train_stats = train_for_an_epoch(
            model, criterion, dataloader_train, optimizer, epoch, epochs,
            log_interval=log_interval, fp16_scaler=fp16_scaler)
        val_stats = test_for_an_epoch(
            model, criterion, dataloader_validation, epoch, epochs,
            no_use_several_test_samples=no_use_several_test_samples,
            name="Val")
        stats = {
            "epoch": epoch,
            "num_tr": train_stats["loss"][1],
            "tr_loss": train_stats["loss"][0],
            "tr_acc": train_stats["prec1"][0],
            "num_va": val_stats["loss"][1],
            "va_loss": val_stats["loss"][0],
            "va_acc": val_stats["prec1"][0],
            "best_va_loss": best_val_loss,
            "best_va_epoch": best_epoch
        }
        if not no_test_set_eval:
            te_rw_stats = test_for_an_epoch(
                model, criterion, dataloader_test_rw, epoch, epochs,
                no_use_several_test_samples=no_use_several_test_samples,
                name="Test (Rwanda)")
            te_ug_stats = test_for_an_epoch(
                model, criterion, dataloader_test_ug, epoch, epochs,
                no_use_several_test_samples=no_use_several_test_samples,
                name="Test (Uganda)")
            stats.update({
                "num_te_rw": te_rw_stats["loss"][1],
                "te_rw_loss": te_rw_stats["loss"][0],
                "te_rw_acc": te_rw_stats["prec1"][0],
                "num_te_ug": te_ug_stats["loss"][1],
                "te_ug_loss": te_ug_stats["loss"][0],
                "te_ug_acc": te_ug_stats["prec1"][0],
            })
        else:
            print("Test sets are not tested.")
        stats_dict.append(stats)
        loss_val = val_stats["loss"][0]
        if scheduler is not None:
            scheduler.step(loss_val)

        if loss_val < best_val_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": loss_val,
                    "val_acc": val_stats["prec1"][0],
                },
                best_save_fp,
            )
            best_val_loss = loss_val
            best_epoch = epoch
            print("Save to {}".format(best_save_fp))
        else:
            print("Model can be found under", args.save_dir)

    with open(path.join(save_dir, "logs.json"), "w+") as f:
        json.dump(stats_dict, f, indent=4)


def main(args):
    # init distributed mode
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    utils.print_args(args)

    if utils.is_main_process():
        if not path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
            print("Creating", args.save_dir)
        args.finished = False
        with open(path.join(args.save_dir, "opts.json"), "w+") as f:
            json.dump(vars(args), f, indent=4)

    # model
    num_channels = get_num_channels(args.data_modalities)
    model = initialize_model(args.model, 2, num_channels, 
                             use_last_n_layers=args.use_last_n_layers,
                             use_pretrained=not args.no_use_pretrained)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable params: {}".format(params))
    model = model.cuda()
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # loss
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    dataloaders = get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=not args.no_use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version,
        data_order=args.data_modalities,
        ddp=True
    )

    if args.use_sgd_scheduler:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    train(
        model,
        criterion,
        dataloaders,
        optimizer,
        args.epochs,
        args.save_dir,
        log_interval=args.log_interval,
        scheduler=scheduler,
        no_use_several_test_samples=args.no_use_several_test_samples,
        no_test_set_eval=args.no_test_set_eval,
        fp16_scaler=fp16_scaler
    )

    args.finished = True
    if utils.is_main_process():
        with open(path.join(args.save_dir, "opts.json"), "w+") as f:
            json.dump(vars(args), f, indent=4)


def evaluate(args_main):
    opts_fp = path.join(args_main.save_dir, "opts.json")
    logs_fp = path.join(args_main.save_dir, "logs.json")
    if not (path.isfile(opts_fp) and path.isfile(logs_fp)):
        print("{} is not finished.".format(args_main.save_dir))
        return
    
    with open(path.join(args_main.save_dir, "opts.json")) as f:
        opts = json.load(f)
    args = Args(opts)

    args.no_use_several_test_samples = args_main.no_use_several_test_samples
    args.num_test_samples = args_main.num_test_samples
    args.test_batch_size = args_main.test_batch_size
    output_file = path.join(args_main.save_dir, "stats_{}_{}.json".format(
        args.no_use_several_test_samples, args.num_test_samples))
    # if path.isfile(output_file):
    #     print("{} already exists.".format(output_file))
    #     return 
    print("Evaluate {} for no_use_several_test_samples: {}".format(
        args.save_dir, args.no_use_several_test_samples) + 
        " num_test_samples: {} test_batch_size: {}".format(
            args.num_test_samples, args.test_batch_size))
    print("Output will be saved to {}".format(output_file))
    # model
    num_channels = get_num_channels(args.data_modalities)
    model = initialize_model(args.model, 2, num_channels, 
                             use_last_n_layers=args.use_last_n_layers,
                             use_pretrained=not args.no_use_pretrained)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable params: {}".format(params))

    best_model_fp = path.join(args.save_dir, "model_best.pt")
    checkpoint = torch.load(best_model_fp, map_location="cpu")
    state_dict = checkpoint['model_state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    print("Loaded model from epoch {} with val loss {} and val acc {}".format(
        checkpoint["epoch"], round(checkpoint["val_loss"], 4), 
        round(checkpoint["val_acc"], 2)
    ))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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
    (_, dataloader_validation, dataloader_test, dataloader_test_rw,
     dataloader_test_ug, _) = dataloaders
    dataloader_tuples = [
        (dataloader_validation, "val"),
        (dataloader_test, "test"),
        (dataloader_test_rw, "test_rw"),
        (dataloader_test_ug, "test_ug")
    ]
    stats = {}
    for dataloader, name in dataloader_tuples:
        all_preds = []
        all_gt = []
        running_loss = 0.
        running_num = 0.
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
                if not args.no_use_several_test_samples:
                    batch_size, num_samples, c, w, h = inputs.shape
                    inputs = inputs.view(batch_size * num_samples, c, w, h)
                else:
                    batch_size = inputs.shape[0]
                    num_samples = 1

                prediction = model(inputs)

                if not args.no_use_several_test_samples:
                    labels_ = labels.unsqueeze(-1).repeat(
                        1, num_samples).view(-1)
                    loss = criterion(prediction, labels_)
                else:
                    loss = criterion(prediction, labels)
                if not args.no_use_several_test_samples:
                    prediction = torch.softmax(
                        prediction.view(
                            batch_size, num_samples, -1), -1).mean(1)
                running_loss += loss.item() * batch_size
                running_num += batch_size
                all_preds.append(prediction.argmax(1).cpu())
                all_gt.append(labels.cpu())
            all_preds = torch.cat(all_preds).numpy()
            all_gt = torch.cat(all_gt).numpy()
            stats["{}_{}".format(name, "acc")] = balanced_accuracy_score(
                all_gt, all_preds)
            stats["{}_{}".format(name, "weighted_f1")] = f1_score(
                all_gt, all_preds, average="weighted")
            stats["{}_{}".format(name, "loss")] = running_loss / running_num
    print(json.dumps(stats, indent=4))
    with open(output_file, "w+") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # model
    parser.add_argument("--model", type=str,
                        choices=["resnet50", "resnet18", "wide_resnet50_2",
                                 "efficientnet_v2_s", "efficientnet_v2_m"],
                        default="resnet50")

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
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--no_test_set_eval", action="store_true")
    parser.add_argument("--use_last_n_layers", default=-1, type=int)
    parser.add_argument("--no_use_pretrained", action="store_true")

    # log during training
    parser.add_argument("--log_interval", default=1, type=int,
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
    # distributed run
    parser.add_argument("--dist_url", default="env://", type=str,
                        help=("url used to set up distributed training; see "
                              "https://pytorch.org/docs/stable/"
                              "distributed.html"))
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    if args.evaluate:
        evaluate(args)
    else:
        main(args)
