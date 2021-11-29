import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src import models
from src.utils import ddp


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_options()
    print("git:\n  {}\n".format(ddp.get_sha()))
    print("Args:")
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    # init distributed mode
    cuda = torch.cuda.is_available()
    if cuda:
        ddp.init_distributed_mode(args)
    ddp.fix_random_seeds(args.seed, cuda=cuda)

    cudnn.benchmark = True

    model = models.BridgeResnet(args.in_channels, model_name=args.model)
    if ddp.has_batchnorms(model) and cuda:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cuda:
        model = model.cuda()
