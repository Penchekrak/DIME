import math
import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def point_on_line(start, end, t):
    return (1 - t) * start + t * end


class Curve(nn.Module):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 requires_grad: bool = True,
                 freeze_start: bool = False,
                 freeze_end: bool = False) -> None:
        super(Curve, self).__init__()
        self.start = nn.Parameter(start, requires_grad=not freeze_start)
        self.end = nn.Parameter(end, requires_grad=not freeze_end)
        self.requires_grad = requires_grad

    def get_point(self, t: float) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t_batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.get_point(t) for t in t_batch])


class Polyline2(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 point: torch.Tensor = None,
                 **curve_kwargs) -> None:
        super(Polyline2, self).__init__(start, end, **curve_kwargs)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point, requires_grad=self.requires_grad)

    def get_point(self, t: float) -> torch.Tensor:
        if t < .5:
            return point_on_line(self.start, self.point, 2 * t)
        return point_on_line(self.point, self.end, 2 * t - 1)

    def inner_parameters(self) -> tp.List[torch.Tensor]:
        return [self.point]


class PolylineN(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 n_nodes: int = 1,
                 **curve_kwargs) -> None:
        super(PolylineN, self).__init__(start, end, **curve_kwargs)
        self.n_nodes = n_nodes

        self.points = []
        for i in range(1, n_nodes):
            self.points.append(Parameter(point_on_line(start, end, i / n_nodes),
                                    requires_grad=self.requires_grad))
        self.params = nn.ParameterList(self.points)
        self._segments = [self.start] + self.points + [self.end]

    def get_point(self, t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            start_ix = torch.trunc(self.n_nodes * t).int()
            end_ix = torch.ceil(self.n_nodes * t).int()
        else:
            start_ix = math.trunc(self.n_nodes * t)
            end_ix = math.ceil(self.n_nodes * t)
        return point_on_line(self._segments[start_ix],
                             self._segments[end_ix],
                             t * self.n_nodes - start_ix)

    def inner_parameters(self) -> tp.List[torch.Tensor]:
        return self.points


class QuadraticBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 point: torch.Tensor = None,
                 **curve_kwargs) -> None:
        super(QuadraticBezier, self).__init__(start, end, **curve_kwargs)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point, requires_grad=self.requires_grad)

    def get_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 2 * self.start + \
               2 * t * (1 - t) * self.point + \
               (1 - t) ** 2 * self.end

    def inner_parameters(self) -> tp.List[torch.Tensor]:
        return [self.point]


class CubicBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 **curve_kwargs) -> None:
        super(CubicBezier, self).__init__(start, end, **curve_kwargs)
        p1 = p1 or point_on_line(start, end, .33)
        p2 = p2 or point_on_line(start, end, .67)
        self.p1 = Parameter(p1, requires_grad=self.requires_grad)
        self.p2 = Parameter(p2, requires_grad=self.requires_grad)

    def get_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 3 * self.start + \
               3 * t * (1 - t) ** 2 * self.p1 + \
               3 * t ** 2 * (1 - t) * self.p2 + \
               (1 - t) ** 3 * self.end

    def inner_parameters(self) -> tp.List[torch.Tensor]:
        return [self.p1, self.p2]


class StateDictCurve(nn.Module):
    frozen_params = ["running_mean",
                     "running_var"]

    def __init__(self,
                 start: tp.OrderedDict,
                 end: tp.OrderedDict,
                 curve_type: tp.ClassVar[Curve],
                 **curve_kwargs):
        super().__init__()
        curves: tp.OrderedDict[str, Curve] = OrderedDict()
        for param_name in start:
            _, param_type = param_name.rsplit(".", 1)
            require_grad = param_type not in self.frozen_params
            curves[param_name.replace(".", "-")] = curve_type(start[param_name],
                                                              end[param_name],
                                                              requires_grad=require_grad,
                                                              **curve_kwargs)
        self.curves = nn.ModuleDict(curves)

    def start_parameters(self):
        return [curve.start for curve in self.curves.values()]

    def inner_parameters(self):
        return [param for curve in self.curves.values() for param in curve.inner_parameters() ]

    def end_parameters(self):
        return [curve.end for curve in self.curves.values()]

    def get_point(self, t):
        return OrderedDict([(param_name.replace("-", "."), curve.get_point(t))
                            for param_name, curve in self.curves.items()])

