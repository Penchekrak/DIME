import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_grad(weights: tp.OrderedDict) -> None:
    for param_name, param in weights.items():
        weights[param_name] = nn.Parameter(param)


def distance(state_dict1: tp.OrderedDict, state_dict2: tp.OrderedDict) -> torch.Tensor:
    distances = []
    for param1, param2 in zip(state_dict1.values(), state_dict2.values()):
        distances.append(F.mse_loss(param1, param2))
    return torch.stack(distances).sum()


def to_tensor(state_dict: tp.OrderedDict) -> tp.Tuple[torch.Tensor, tp.OrderedDict]:
    sizes = OrderedDict()
    weights = []
    for param_name, param in state_dict.items():
        sizes[param_name] = (param.numel(), param.size())
        weights.append(param.flatten())
    return torch.hstack(weights), sizes


def to_state_dict(weights: torch.Tensor, sizes: tp.OrderedDict) -> tp.OrderedDict:
    state_dict = OrderedDict()
    offset = 0
    for param_name, shape_data in sizes.items():
        param_numel, param_size = shape_data
        state_dict[param_name] = weights[offset:offset + param_numel].view(param_size)
        offset += param_numel
    return state_dict

def to_device(state_dict: tp.OrderedDict, device) -> None:
    for param_name, param in state_dict.items():
        state_dict[param_name] = param.to(device)
