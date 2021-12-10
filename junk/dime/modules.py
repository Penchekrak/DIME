import itertools
import typing as tp

import torch
from torch import nn


class Evaluator(nn.Module):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def forward(self, weights: tp.Mapping[str, torch.Tensor], x: tp.Union[tp.Tuple[torch.Tensor], torch.Tensor]):
        self.model.set_weights(weights)
        return self.model(x)


class WeightFreeModule(nn.Module):
    def set_weights(self, prefix: str, state_dict: tp.Mapping[str, torch.Tensor]):
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            param = state_dict[key]

        for name, child in self._modules.items():
            if child is not None and isinstance(child, WeightFreeModule):
                child.set_weights(prefix + name + '.', state_dict)

class SettableParameter()