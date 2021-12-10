import typing as tp

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


kwarg_names = {nn.Linear: tuple(),
               nn.Conv2d: ("stride",
                           "padding",
                           "dilation",
                           "groups"),
               nn.BatchNorm2d: ("eps",
                                "momentum"),
               nn.ReLU: ("inplace",),
               nn.Dropout: ("p",)}


def get_kwargs(layer: nn.Module) -> tp.Dict[str, tp.Any]:
    """Extracts the neccessary parameters from a `layer` instance to pass to a
    corresponding function.

    Args:
        layer (Module): The layer of type present in `kwarg_names`.

    Returns:
        Dict[str, Any]: kwargs for the layer, exact list is given in
        `kwarg_names`.
    """
    kw_names = kwarg_names.get(type(layer), tuple())
    return {kw_name: getattr(layer, kw_name)
            for kw_name in kw_names}


def get_layer_kwargs(model: nn.Module) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """Builds a dict constaning kwargs for each layer of the `model`. The dict
    structure is the same as in `model.named_modules()`.

    Args:
        model (nn.Module): An instance of NN.

    Returns:
        Dict[str, tp.Dict[str, Any]]: A dict of kwargs for every layer.
    """
    return {module_name: get_kwargs(module)
            for module_name, module in model.named_modules()}


def get_children(layer_name: str, layer: nn.Module) -> tp.List[str]:
    """Lists the immediate children of the layer

    Args:
        layer (nn.Module): The layer of a model.
        prefix (str, optional): [description]. Defaults to "".


    Returns:
        tp.List[str]: A list of children names.
    """
    if not layer_name:
        return [name for name, _ in layer.named_children()]
    return [f"{layer_name}.{name}" for name, _ in layer.named_children()]


def get_model_structure(model: nn.Module) -> tp.Dict[str, tp.List[str]]:
    """Generates a dict with the children for each layer in `model`.

    Args:
        model (nn.Module): An instance of NN.

    Returns:
        tp.Dict[str, tp.List[str]]: A dict of kwargs for every layer.
    """
    return {module_name: get_children(module_name, module)
            for module_name, module in model.named_modules()}


def format_state_dict(state_dict: tp.OrderedDict[str, torch.Tensor]) -> \
    tp.OrderedDict[str, tp.Dict[str, torch.Tensor]]:
    """Groups the parameters in `state_dict` by their module.

    Args:
        state_dict OrderedDict[str, Tensor]): A `model.state_dict()` instance.

    Returns:
        OrderedDict[str, Dict[str, Tensor]]: The formatted state_dict.
    """
    formatted_dict = OrderedDict()
    for param_name, param in state_dict.items():
        module_name, param_name = param_name.rsplit(".", 1)
        formatted_dict.setdefault(module_name, {})
        formatted_dict[module_name][param_name] = param
    return formatted_dict


class FunctionalNet:
    def __init__(self, model):
        self._build_layer_funcs(model)
        self._layer_kwargs = get_layer_kwargs(model)
        self._model_structure = get_model_structure(model)

        self.state_dict = {}

    def _build_layer_funcs(self, model):
        self._functionals = {nn.Linear: F.linear,
                             nn.Conv2d: F.conv2d,
                             nn.BatchNorm2d: F.batch_norm,
                             nn.ReLU: F.relu,
                             nn.Dropout: F.dropout,
                             nn.Sequential: self.apply_sequential}

        self._layer_funcs = {module_name: self._functionals[type(module)]
                             for module_name, module in model.named_modules()}

    def __call__(self, input, weights):
        self.state_dict = format_state_dict(weights)
        return self.apply_layer("", input)

    def apply_layer(self, layer_name, input):
        func = self._layer_funcs[layer_name]
        children = self._model_structure[layer_name]
        kwargs = self._layer_kwargs[layer_name]
        params = self.state_dict.get(layer_name, {})

        return func(input, *children, **params, **kwargs)

    def apply_sequential(self, input, *children):
        for child in children:
            input = self.apply_layer(child, input)
        return input


def example():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    x = torch.ones(1, 2)
    print(model(x))
    f_model = FunctionalNet(model)
    weights = format_state_dict(model.state_dict())
    print(f_model(x, weights))
