"""
Code adjusted from https://github.com/YU1ut/MixMatch-pytorch/train.py
"""

import argparse
import json
import os
import shutil
import time

import os.path as path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

from src.utils import AverageMeter, accuracy
from sklearn.metrics import balanced_accuracy_score, f1_score

from src import models, utils
from src.data import bridge_site

from train_baseline import Args

VAL_LOG_FORMAT = (
    '{name} ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '
    'Total: {total:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f}')
TRAIN_LOG_FORMAT = (
    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '
    'Total: {total:.3f}s | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | '
    'Loss_u: {loss_u:.4f} | W: {w:.4f}')


def main(args):
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()

    # Random seed
    utils.fix_random_seeds(args.manualSeed)
    if os.path.isdir(args.out):
        print("Save dir {} already exists.".format(args.out))
        return
    else:
        os.makedirs(args.out)

    dataloaders = bridge_site.get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=args.use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version,
        data_order=args.data_modalities
    )

    (labeled_trainloader, val_loader, _, _, _, 
     unlabeled_trainloader) = dataloaders

    # model
    num_channels = bridge_site.get_num_channels(args.data_modalities)
    model = models.initialize_mod_model(args.model, 2, num_channels, 
                             use_last_n_layers=args.use_last_n_layers,
                             use_pretrained=not args.no_use_pretrained).cuda()
    ema_model = models.initialize_mod_model(args.model, 2, num_channels, 
                             use_last_n_layers=args.use_last_n_layers,
                             use_pretrained=not args.no_use_pretrained).cuda()
    for param in ema_model.parameters():
        param.detach_()

    cudnn.benchmark = True
    print('Total params: {:.2f}M'.format(sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, args.lr, alpha=args.ema_decay)
    start_epoch = 0

    columns = ['Epoch', 'Train Loss', 'Train Loss X', 'Train Loss U',
               'Train Acc.', 'Val Loss', 'Val Acc.', 'Best Epoch',
               'Best Acc.']
    logs = []
    best_acc = best_epoch = -1
    args.finished = False

    with open(os.path.join(args.out, "opts.json"), "w+") as f:
        json.dump(vars(args), f, indent=4)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss, train_loss_x, train_loss_u = train(
            labeled_trainloader, unlabeled_trainloader, model, optimizer,
            ema_optimizer, train_criterion, epoch, use_cuda, args.T,
            args.alpha, args.lambda_u)
        _, train_acc = validate(
            labeled_trainloader, ema_model, criterion, use_cuda,
            mode='Train Stats')
        val_loss, val_acc = validate(
            val_loader, ema_model, criterion, use_cuda, mode='Val Stats',
            use_several_test_samples=args.use_several_test_samples)
        # test_loss, test_acc = validate(
        #     test_loader, ema_model, criterion, use_cuda, mode='Test Stats ',
        #     use_several_test_samples=args.use_several_test_samples)

        # append logger file
        logs.append([int(epoch), float(np.round(train_loss, 4)),
                     float(np.round(train_loss_x, 4)),
                     float(np.round(train_loss_u, 4)),
                     float(np.round(train_acc, 4)),
                     float(np.round(val_loss, 4)),
                     float(np.round(val_acc, 4)),
                     float(np.round(best_epoch, 4)),
                     float(np.round(best_acc, 4))])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out)
        print('Epoch: [{:d} | {:d}] LR: {:f} Total: {:.3f}s\n'.format(
            epoch + 1, args.epochs, state['lr'], time.time() - start))
        if epoch - best_acc > 50 and epoch > 150 and not args.no_early_stopping:
            print("Early stopping...")
            break
    pd.DataFrame(logs, columns=columns).to_csv(
        path.join(args.out, "logs.csv"), index=False)
    args.finished = True

    with open(os.path.join(args.out, "opts.json"), "w+") as f:
        json.dump(vars(args), f, indent=4)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
          ema_optimizer, criterion, epoch, use_cuda, T, alpha, lambda_u):
    start = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    unlabeled_train_iter = iter(unlabeled_trainloader)

    def log():
        print(TRAIN_LOG_FORMAT.format(
            total=time.time() - start,
            batch=batch_idx + 1,
            size=len(labeled_trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        ))

    model.train()
    for batch_idx, (inputs_x, targets_x) in enumerate(labeled_trainloader):
        inputs_u12 = unlabeled_train_iter.next()
        inputs_u, inputs_u2 = torch.split(inputs_u12, 1, dim=1)
        inputs_x = inputs_x.float()
        inputs_u = inputs_u.squeeze(1).float()
        inputs_u2 = inputs_u2.squeeze(1).float()
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 2).scatter_(
            1, targets_x.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) +
                 torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1 / T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct
        # batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u,
                              mixed_target[batch_size:],
                              epoch * len(labeled_trainloader) + batch_idx,
                              lambda_u)
        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            log()
    log()

    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(dataloader, model, criterion, use_cuda, mode,
             use_several_test_samples=False):
    start = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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
            if use_cuda:
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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    log()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    print("[Save] Save model to {}".format(filepath))
    torch.save(state, filepath)
    if is_best:
        best_fp = os.path.join(checkpoint, 'model_best.pth.tar')
        print("[Save] Save best model to {}".format(best_fp))
        shutil.copyfile(filepath, best_fp)


def linear_rampup(current, rampup_length=1000):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u,
                 current_iter, lambda_u):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                                                 dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(current_iter)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def evaluate(args_main):
    opts_fp = path.join(args_main.out, "opts.json")
    if not path.isfile(opts_fp):
        print("{} is not finished.".format(args_main.out))
        return
    with open(path.join(args_main.out, "opts.json")) as f:
        opts = json.load(f)
    args = Args(opts)
    
    args.use_several_test_samples = args_main.use_several_test_samples
    args.num_test_samples = args_main.num_test_samples
    args.test_batch_size = args_main.test_batch_size
    if args_main.use_last_checkpoint:
        model_fp = path.join(args.out, "checkpoint.pth.tar")
    else:
        model_fp = path.join(args.out, "model_best.pth.tar")
    if not path.isfile(model_fp):
        print("{} does not exists.".format(model_fp))
        return
    checkpoint = torch.load(model_fp, map_location="cpu")
    best_epoch = checkpoint["epoch"]

    output_file = path.join(args_main.out, "stats_{}_{}_{}.json".format(
        "last" if args_main.use_last_checkpoint else "best",
        args.use_several_test_samples, args.num_test_samples))
    
    if path.isfile(output_file):
        with open(output_file) as f:
            stats = json.load(f)
        if args_main.use_last_checkpoint:
            if best_epoch - stats["epoch"] < 5 and best_epoch != 200:
                print("Only calculate last epoch every 5 epochs.")
                return
        else:
            if stats["epoch"] == best_epoch and not args_main.overwrite:
                print("Stats for epoch {} already exists.".format(best_epoch))
                return
    
    print("Evaluate {} for use_several_test_samples: {}".format(
        args.out, args.use_several_test_samples) + 
        " num_test_samples: {} test_batch_size: {}".format(
            args.num_test_samples, args.test_batch_size))
    print("Output will be saved to {}".format(output_file))

    use_cuda = torch.cuda.is_available()
    num_channels = bridge_site.get_num_channels(args.data_modalities)
    ema_model = models.initialize_mod_model(
        args.model, 2, num_channels, use_last_n_layers=args.use_last_n_layers,
        use_pretrained=not args.no_use_pretrained)
    if use_cuda:
        ema_model = ema_model.cuda()
    ema_model.eval()
    
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    print(
        "Loaded model from epoch {epoch} ".format(epoch=checkpoint["epoch"]) +
        "with validation accuracy {acc:.2f}".format(
            acc=checkpoint["best_acc"]))
    
    dataloaders = bridge_site.get_dataloaders(
        args.batch_size, args.tile_size,
        use_augment=not args.no_augmentation,
        use_several_test_samples=args.use_several_test_samples,
        num_test_samples=args.num_test_samples,
        test_batch_size=args.test_batch_size,
        data_version=args.data_version,
        data_order=args.data_modalities
    )

    (_, val_loader, test_loader, test_rw_loader, test_ug_loader, 
     _) = dataloaders

    loaders = [("val", val_loader), ("test", test_loader), 
               ("test_rw", test_rw_loader), ("test_ug", test_ug_loader)]
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    stats = {"epoch": best_epoch}
    for name, dataloader in loaders:
        all_preds = []
        all_gt = []
        running_loss = 0.
        running_num = 0.
        with torch.no_grad():
            for inputs, targets in dataloader:
                # measure data loading time
                inputs = inputs.float()
                batch_size = inputs.shape[0]
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if args.use_several_test_samples:
                    batch_size, num_samples, c, w, h = inputs.shape
                    inputs_ = inputs.view(batch_size * num_samples, c, w, h)
                    targets_ = targets.view(-1, 1).repeat(1, num_samples).view(
                        batch_size * num_samples)
                else:
                    inputs_ = inputs
                    targets_ = targets
                # compute output
                outputs = ema_model(inputs_)
                loss = criterion(outputs, targets_)
                # measure accuracy and record loss
                if args.use_several_test_samples:
                    outputs = torch.softmax(
                        outputs.view(batch_size, num_samples, -1), -1).mean(1)
                    loss = loss.view(batch_size, num_samples).mean(1).mean()
                else:
                    loss = loss.mean()
                running_loss += loss.item() * batch_size
                running_num += batch_size
                all_preds.append(outputs.argmax(1).cpu())
                all_gt.append(targets.cpu())
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
    # Optimization options
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0,
                        help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # Method options
    parser.add_argument('--train-iteration', type=int, default=256,
                        help='Number of iteration per epoch')
    parser.add_argument('--out', default='experiments/ssl',
                        help='Directory to output the result')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=5, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.75, type=float)
    parser.add_argument("--tile_size", type=int,
                        choices=[300, 600, 1200], default=1200)
    # model
    parser.add_argument("--model", type=str,
                        choices=["resnet50", "resnet18", "wide_resnet50_2",
                                 "efficientnet_v2_s", "efficientnet_v2_m"],
                        default="resnet50")

    parser.add_argument("--use_several_test_samples", action="store_true")
    parser.add_argument("--num_test_samples", default=32, type=int)
    parser.add_argument("--test_batch_size", default=6, type=int)
    parser.add_argument("--data_version", default="v1", type=str)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--use_last_n_layers", default=-1, type=int)
    parser.add_argument("--data_modalities", nargs="+", type=str,
                        default=["population", "osm_img", "elevation",
                                 "slope", "roads", "waterways",
                                 "admin_bounds_qgis"])
    parser.add_argument("--no_use_pretrained", action="store_true")
    parser.add_argument("--no_early_stopping", action="store_true")

    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_last_checkpoint", action="store_true")

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        main(args)
