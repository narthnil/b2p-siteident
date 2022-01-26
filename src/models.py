import torch
import torch.nn as nn
import torchvision.models as models


class BridgeResnet(nn.Module):

    def __init__(self, model_name: str = "resnet18",
                 pretrained: bool = True, lazy: bool = True,
                 num_channels: int = 9) -> None:
        super().__init__()
        assert model_name in ["resnet18", "resnet50", "resnext",
                              "efficientnet_b2", "efficientnet_b7"], \
            "Model {} not known.".format(model_name)

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnext":
            self.model = models.resnext101_32x4d(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == "efficientnet_b7":
            self.model = models.efficientnet_b2(pretrained=pretrained)
        else:
            raise NotImplementedError
        if lazy and not model_name.startswith("efficientnet"):
            self.model.conv1 = nn.LazyConv2d(
                64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        elif not lazy and not model_name.startswith("efficientnet"):
            self.model.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        elif not lazy and model_name == "efficientnet_b2":
            self.model.features[0][0] = nn.Conv2d(
                num_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 2)
        elif not lazy and model_name == "efficientnet_b7":
            self.model.features[0][0] = nn.Conv2d(
                num_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                padding=(1, 1), bias=False)
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 2)
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        return output
