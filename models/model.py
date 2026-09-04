"""Transfer-learning model definition."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes: int = 37,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create ResNet-18 and replace its ImageNet classifier for 37 breeds."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

