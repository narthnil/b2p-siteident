import torch.nn as nn

from torchvision import models


NUM_LAYERS = {
    "resnet18": 52,
    "resnet50": 126,
    "wide_resnet50_2": 126,
    "efficientnet_v2_s": 485,
    "efficientnet_v2_m": 697,
}


def count_num_layers(m: nn.Module):
    stack = list(m.children())
    num_layers = 0
    while stack:
        next_elem = stack.pop(0)
        children = list(next_elem.children())
        if len(children) == 0:
            num_layers += 1
        else:
            stack = children + stack
    return num_layers


def set_parameter_requires_grad(model, model_name, use_last_n_layers):
    if use_last_n_layers == -1:
        return
    current_count = 0
    stack = list(model.children())
    while stack:
        current_elem = stack.pop(0)
        children = list(current_elem.children())
        if len(children) == 0:
            current_count += 1
            if current_count < NUM_LAYERS[model_name] - use_last_n_layers + 1:
                for param in current_elem.parameters():
                    param.requires_grad = False
        else:
            stack = children + stack


def initialize_mod_model(model_name, num_classes, num_channels, 
                         use_last_n_layers=-1, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each
    # of these variables is model specific
    if use_pretrained:
        if model_name.startswith("efficientnet") or model_name == "resnet18":
            weights = "IMAGENET1K_V1"
        else:
            weights = "IMAGENET1K_V2"
    else:
        weights = None
    model = getattr(models, model_name)(weights=weights)
    set_parameter_requires_grad(model, model_name, use_last_n_layers)
    if model_name.startswith("efficientnet"):
        model.features[0][0] = nn.Conv2d(
                num_channels, model.features[0][0].out_channels, 
                kernel_size=model.features[0][0].kernel_size, 
                stride=model.features[0][0].stride,
                padding=model.features[0][0].padding, 
                bias=model.features[0][0].bias)
        for param in model.features.parameters():
            param.requires_grad = True
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes))
    else:
        model.conv1 = nn.Conv2d(
            num_channels, model.conv1.out_channels, 
            kernel_size=model.conv1.kernel_size, 
            stride=model.conv1.stride)
        for param in model.conv1.parameters():
            param.requires_grad = True
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def initialize_rgb_model(model_name, num_classes, use_last_n_layers=1,
                         use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each
    # of these variables is model specific
    if use_pretrained:
        if model_name.startswith("efficientnet") or model_name == "resnet18":
            weights = "IMAGENET1K_V1"
        else:
            weights = "IMAGENET1K_V2"
    else:
        weights = None
    model = getattr(models, model_name)(weights=weights)
    set_parameter_requires_grad(model, model_name, use_last_n_layers)
    if model_name.startswith("efficientnet"):
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes))
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model
