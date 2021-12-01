import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src import models
from src.utils import dist


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Argument for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # model
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Number of input data channels")
    parser.add_argument("--model", type=str, choices=["resnet18"],
                        default="resnet18")

    # dist
    parser.add_argument("--dist_url", default="env://", type=str,
                        help=("url used to set up distributed training; see "
                              "https://pytorch.org/docs/stable/distributed.html"))
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")

    # log during training
    parser.add_argument("--log_interval", default=None, type=int,
                        help=("Whether to log every `log_interval`-th "
                              "iterations."))

    # training
    parser.add_argument("--epochs", default=300, type=int,
                        help="Training epochs.")

    # optimizer
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="Adam optimizer learning rate.")
    args = parser.parse_args()
    return args


def train(model: nn.Module, criterion: nn.modules.loss._Loss,
          dataloader, optimizer, epoch: int,
          cuda: bool = True, log_interval: int = None):

    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        if log_interval is not None and i % log_interval == 0:
            pass
    print("==> Train")


def test(model: nn.Module, dataloader, epoch: int):
    print("==> Test")


def run(model: nn.Module, criterion: nn.Module, dataloaders, optimizer,
        epochs: int, log_interval: int = None):
    print("==> Start training")


if __name__ == "__main__":
    args = parse_options()
    print("git:\n  {}\n".format(dist.get_sha()))
    print("Args:")
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    # init distributed mode
    cuda = torch.cuda.is_available()
    if cuda:
        dist.init_distributed_mode(args)
    dist.fix_random_seeds(args.seed, cuda=cuda)

    cudnn.benchmark = True
    # model
    model = models.BridgeResnet(args.in_channels, model_name=args.model)
    # loss
    criterion = nn.CrossEntropyLoss(reduction="none")
    # ddp, cuda
    if dist.has_batchnorms(model) and cuda:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # TODO: get dataloaders (train, val, test)
    dataloaders = None

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run(model, criterion, dataloaders, optimizer, args.epochs,
        log_interval=args.log_interval)
