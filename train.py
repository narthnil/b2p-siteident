import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src import models
from src.utils import argparser, dist
from src.data import get_dataloaders


def train_for_an_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: DataLoader,
    optimizer,
    epoch: int,
    cuda: bool = True,
    log_interval: int = None,
):
    print(f"Starting training pass over epoch {epoch}")
    model.train()
    epoch_loss = 0
    num_train = 0
    acc = 0
    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

        prediction = model(inputs)
        loss = criterion(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc += (torch.argmax(prediction, -1) == labels).sum()
        epoch_loss += loss.item() * inputs.shape[0]
        num_train += inputs.shape[0]
        if log_interval is not None and i % log_interval == 0:
            print(f"Step loss: {loss.item()}")
    return epoch_loss / num_train, acc / num_train


def test_for_an_epoch(model: nn.Module, criterion: nn.modules.loss._Loss,
                      dataloader, epoch: int, cuda: bool):
    print(f"Starting validation pass over epoch {epoch}")
    model.eval()
    epoch_loss = 0
    num_test = 0
    acc_test = 0
    for i, (inputs, labels) in enumerate(dataloader):
        if cuda:
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()

        prediction = model(inputs)
        loss = criterion(prediction, labels)
        acc_test += (torch.argmax(prediction, -1) == labels).sum()
        epoch_loss += loss.item() * inputs.shape[0]
        num_test += inputs.shape[0]
    return epoch_loss / num_test, acc_test / num_test


def run(
    model: nn.Module,
    criterion: nn.Module,
    dataloaders,
    optimizer,
    epochs: int,
    log_interval: int = None,
    cuda: bool = True
):
    dataloader_train, dataloader_validation, dataloader_test = dataloaders
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        epoch_loss_train, acc_train = train_for_an_epoch(
            model,
            criterion,
            dataloader_train,
            optimizer,
            epoch,
            log_interval=log_interval,
            cuda=cuda
        )
        with torch.no_grad():
            epoch_loss_validation, acc_val = test_for_an_epoch(
                model, criterion, dataloader_validation, epoch, cuda)
            epoch_loss_test, acc_test = test_for_an_epoch(
                model, criterion, dataloader_test, epoch, cuda)
        print(f"Training epoch loss for epoch {epoch}: {epoch_loss_train}")
        print(f"Training epoch acc for epoch {epoch}: {acc_train}")
        print(
            f"Validation epoch loss for epoch {epoch}: {epoch_loss_validation}")
        print(
            f"Validation epoch acc for epoch {epoch}: {acc_val}")
        print(
            f"Test epoch loss for epoch {epoch}: {epoch_loss_test}")
        print(
            f"Test epoch acc for epoch {epoch}: {acc_test}")
    return model


def train():
    args = argparser.get_args()
    print("git:\n  {}\n".format(dist.get_sha()))
    print("Args:")
    print(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )
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
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    dataloaders = get_dataloaders(
        args.batch_size, args.tile_size
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trained_model = run(
        model,
        criterion,
        dataloaders,
        optimizer,
        args.epochs,
        log_interval=args.log_interval,
        cuda=cuda
    )
    # TODO: Save trained model


if __name__ == "__main__":
    train()
