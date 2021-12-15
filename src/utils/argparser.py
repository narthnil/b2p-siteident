import argparse


def get_args():

    parser = argparse.ArgumentParser("Argument for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # model
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50"],
                        default="resnet18")

    # data
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of samples per training step")
    parser.add_argument("--tile_size", type=int,
                        choices=[300, 600, 1200], default=300)

    # dist
    parser.add_argument("--dist_url", default="env://", type=str,
                        help=(
                            "url used to set up distributed training; see "
                            "https://pytorch.org/docs/stable/distributed.html"
                        ))
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")

    # log during training
    parser.add_argument(
        "--log_interval",
        default=None,
        type=int,
        help=("Whether to log every `log_interval`-th " "iterations."),
    )

    # training
    parser.add_argument("--epochs", default=300, type=int,
                        help="Training epochs.")

    # optimizer
    parser.add_argument("--lr", default=5e-4, type=float,
                        help="Adam optimizer learning rate.")
    args = parser.parse_args()
    return args
