import numpy as np

import torch


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from 
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_args(args):
    """Print out all args argument"""
    print("Args:")
    print(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )


def get_tp_tn_fp_fn(prediction: torch.Tensor, groundtruth: torch.Tensor):
    """Calculates true pos, true neg., false pos. and false neg."""
    tp = torch.logical_and(
        torch.argmax(prediction, -1) == 1, groundtruth == 1).sum()
    fp = torch.logical_and(
        torch.argmax(prediction, -1) == 1, groundtruth == 0).sum()
    tn = torch.logical_and(
        torch.argmax(prediction, -1) == 0, groundtruth == 0).sum()
    fn = torch.logical_and(
        torch.argmax(prediction, -1) == 0, groundtruth == 1).sum()
    return tp.item(), tn.item(), fp.item(), fn.item()


def get_f1(tp, _, fp, fn):
    """Calculates F1 (macro) metric."""
    return tp / (tp + 0.5 * (fp + fn))


def print_confusion_matrix(tp, tn, fp, fn, name=""):
    print("{}confusion matrix".format(name + " "))
    print("{:04d} (TP) {:04d} (FP)".format(tp, fp))
    print("{:04d} (FN) {:04d} (TN)".format(fn, tn))
