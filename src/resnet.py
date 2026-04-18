from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 200, cifar_stem: bool = True) -> nn.Module:
    """Create ResNet-18 for 64x64 Tiny ImageNet images."""

    model = resnet18(weights=None, num_classes=num_classes)
    if cifar_stem:
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
