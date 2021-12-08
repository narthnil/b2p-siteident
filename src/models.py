import torch
import torch.nn as nn
import torchvision.models as models


class BridgeResnet(nn.Module):

    def __init__(self, model_name: str = "resnet18",
                 pretrained: bool = True) -> None:
        super().__init__()
        assert model_name in ["resnet18"], "Model {} not known.".format(
            model_name)

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.model.conv1 = nn.LazyConv2d(
            64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
            bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        return output
