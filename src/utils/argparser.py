import argparse


def get_args(parse_args=True, add_save_dir=True):

    parser = argparse.ArgumentParser("Argument for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # model
    parser.add_argument("--model", type=str,
                        choices=["resnet18", "resnet50", "resnext",
                                 "efficientnet_b2", "efficientnet_b7"],
                        default="efficientnet_b7")

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

    parser.add_argument("--use_several_test_samples", action="store_true")
    parser.add_argument("--num_test_samples", default=16, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)

    # log during training
    parser.add_argument("--log_interval", default=None, type=int,
                        help=("Whether to log every `log_interval`-th "
                              "iterations."))

    # training
    parser.add_argument("--epochs", default=300, type=int,
                        help="Training epochs.")

    # optimizer
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Adam optimizer learning rate.")
    # save directory
    if add_save_dir:
        parser.add_argument("--save_dir", required=True, type=str)
    if parse_args:
        args = parser.parse_args()
        return args
    else:
        return parser
