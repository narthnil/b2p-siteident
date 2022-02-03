from typing import List

import torch


class UnNormalize(object):
    """Undo normalizing a tensor with a mean and standard deviation
    Normalize a Tensor with mean and standard deviation.

    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n
    channels, this transform will undo the normalization of each channel of the
    input torch.*Tensor i.e.,
        output[channel] = input[channel] * std[channel] + mean[channel])
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    """Normalize a Tensor with mean and standard deviation.

    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n
    channels, this transform will normalize each channel of the input
    torch.*Tensor i.e.,
        output[channel] = (input[channel] - mean[channel]) / std[channel]
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        i = 0
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            i += 1
        return tensor
