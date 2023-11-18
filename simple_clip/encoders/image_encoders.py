import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, EfficientNet_V2_S_Weights


class Squeeze(nn.Module):

    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def _prep_encoder(model):
    modules = list(model.children())[:-1]
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(Squeeze())

    return nn.Sequential(*modules)


def resnet18():
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return _prep_encoder(resnet)


def resnet50():
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    return _prep_encoder(resnet)


def efficientnet_v2_s():
    model = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    return _prep_encoder(model)
