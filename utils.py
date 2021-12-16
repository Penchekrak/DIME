import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device_count


def to_parameter(weights: tp.OrderedDict) -> None:
    for param_name, param in weights.items():
        weights[param_name] = nn.Parameter(param)


def distance(state_dict1: tp.OrderedDict, state_dict2: tp.OrderedDict) -> torch.Tensor:
    distances = []
    for param1, param2 in zip(state_dict1.values(), state_dict2.values()):
        distances.append(F.mse_loss(param1, param2, reduction="sum"))
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


def pick_gpus():
    n_devices = device_count()
    if n_devices == 0:
        gpus = None
    elif n_devices == 2:
        # to use only the second GPU on statml3
        gpus = [1]
    else:
        gpus = -1
    return gpus

def unpack_ends(curve_state_dict: tp.OrderedDict) -> tp.Tuple[tp.OrderedDict, tp.OrderedDict]:
    state_dicts = {"start": OrderedDict(), "end": OrderedDict()}
    for param_name, param in curve_state_dict.items():
        _, param_name, kind = param_name.rsplit(".", maxsplit=2)
        if kind not in state_dicts:
            continue
        state_dicts[kind][param_name.replace("-", ".")] = param
    return state_dicts["start"], state_dicts["end"]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, inputs):
        return self.block(inputs)