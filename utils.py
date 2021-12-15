import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_grad(weights: tp.OrderedDict) -> None:
    for param_name, param in weights.items():
        weights[param_name] = nn.Parameter(param)


def distance(weights1: tp.OrderedDict, weights2: tp.OrderedDict) -> torch.Tensor:
    distances = []
    for param1, param2 in zip(weights1.values(), weights2.values()):
        distances.append(F.mse_loss(param1, param2))
    return torch.stack(distances).sum()
