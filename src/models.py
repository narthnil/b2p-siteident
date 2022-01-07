import torch
import torch.nn as nn
import torchvision.models as models


class BridgeResnet(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        assert model_name in ["resnet18", "resnet50"], "Model {} not known.".format(
            model_name
        )

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.model.conv1 = nn.LazyConv2d(
            64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        return output


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, max_pool=False):
        super(CNNBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

        self.pool = max_pool

    def forward(self, _input: torch.Tensor):
        feature_map = self.layer(_input)
        if self.pool:
            feature_map = nn.MaxPool2d(2, 2)(feature_map)
        return feature_map


class ResBlock(nn.Module):
    """
    Single residual layer.
    """

    def __init__(self, in_channels, out_channels, kernel, stride, padding, num_layers, relu=nn.LeakyReLU(0.1), pool=False):
        super(ResBlock, self).__init__()

        self.initial_convolutional_layer = CNNBlock(in_channels, out_channels, kernel, stride, padding, relu)
        self.convolutional_layers = nn.ModuleList(
            [CNNBlock(out_channels, out_channels, kernel, stride, padding, relu) for _ in range(num_layers)]
        )

        self.transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool = pool

    def forward(self, X):
        convolutional_feature_map = self.initial_convolutional_layer(X)
        X_conv_transpose = self.transpose_layer(X)
        residual = X_conv_transpose + convolutional_feature_map
        for layer in self.convolutional_layers:
            convolutional_feature_map = layer(residual)
            residual = residual + convolutional_feature_map

        if self.pool:
            residual = nn.MaxPool2d(2, 2)(residual)

        return residual


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.initial_layers = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            CNNBlock(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), max_pool=True),
            CNNBlock(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), max_pool=True),
            CNNBlock(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), max_pool=True)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, _input: torch.Tensor):
        feature_map = self.initial_layers(_input)
        feature_map = feature_map.reshape((_input.shape[0], -1))
        return self.linear_layers(feature_map)
