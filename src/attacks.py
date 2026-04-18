from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .tiny_imagenet import IMAGENET_MEAN, IMAGENET_STD


def _bounds(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper, std


def _clamp_normalized(images: torch.Tensor) -> torch.Tensor:
    lower, upper, _ = _bounds(images.device)
    return torch.max(torch.min(images, upper), lower)


def fgsm(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Untargeted FGSM. eps is measured in original pixel scale, e.g. 8/255."""

    model.eval()
    _, _, std = _bounds(images.device)
    eps_tensor = torch.tensor(eps, device=images.device).view(1, 1, 1, 1) / std

    adv = images.detach().clone().requires_grad_(True)
    loss = criterion(model(adv), targets)
    grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
    adv = adv + eps_tensor * grad.sign()
    return _clamp_normalized(adv).detach()


def pgd_linf(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    criterion: nn.Module,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted L-infinity PGD. eps and alpha are in original pixel scale."""

    model.eval()
    lower, upper, std = _bounds(images.device)
    eps_tensor = torch.tensor(eps, device=images.device).view(1, 1, 1, 1) / std
    alpha_tensor = torch.tensor(alpha, device=images.device).view(1, 1, 1, 1) / std

    if random_start:
        adv = images + torch.empty_like(images).uniform_(-1.0, 1.0) * eps_tensor
        adv = torch.max(torch.min(adv, upper), lower).detach()
    else:
        adv = images.detach().clone()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = criterion(model(adv), targets)
        grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        adv = adv.detach() + alpha_tensor * grad.sign()
        delta = torch.max(torch.min(adv - images, eps_tensor), -eps_tensor)
        adv = torch.max(torch.min(images + delta, upper), lower).detach()

    return adv
