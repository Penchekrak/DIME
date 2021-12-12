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
                 requires_grad: bool = True) -> None:
        super(Curve, self).__init__()
        self.start = start
        self.end = end
        self.requires_grad = requires_grad

    def get_point(self, t: float) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t_batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.get_point(t) for t in t_batch])


class Polyline2(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 requires_grad: bool = True,
                 point: torch.Tensor = None) -> None:
        super(Polyline2, self).__init__(start, end, requires_grad)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point, requires_grad=self.requires_grad)

    def get_point(self, t: float) -> torch.Tensor:
        if t < .5:
            return point_on_line(self.start, self.point, 2 * t)
        return point_on_line(self.point, self.end, 2 * t - 1)


class PolylineN(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 n_nodes: int,
                 requires_grad: bool = True) -> None:
        super(PolylineN, self).__init__(start, end, requires_grad)
        self.n_nodes = n_nodes

        points = []
        for i in range(1, n_nodes):
            points.append(Parameter(point_on_line(start, end, i / n_nodes),
                                    requires_grad=self.requires_grad))
        self.params = nn.ParameterList(points)
        self._segments = [self.start] + points + [self.end]

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


class QuadraticBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 point: torch.Tensor = None,
                 requires_grad: bool = True) -> None:
        super(QuadraticBezier, self).__init__(start, end, requires_grad)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point, requires_grad=self.requires_grad)

    def get_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 2 * self.start + \
               2 * t * (1 - t) * self.point + \
               (1 - t) ** 2 * self.end


class CubicBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 requires_grad: bool = True) -> None:
        super(CubicBezier, self).__init__(start, end, requires_grad)
        p1 = p1 or point_on_line(start, end, .33)
        p2 = p2 or point_on_line(start, end, .67)
        self.p1 = Parameter(p1, requires_grad=self.requires_grad)
        self.p2 = Parameter(p2, requires_grad=self.requires_grad)

    def get_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 3 * self.start + \
               3 * t * (1 - t) ** 2 * self.p1 + \
               3 * t ** 2 * (1 - t) * self.p2 + \
               (1 - t) ** 3 * self.end


class StateDictCurve:
    frozen_params = ["running_mean",
                     "running_var"]

    def __init__(self, start: OrderedDict, end: OrderedDict, curve_type: tp.ClassVar[Curve], **curve_kwargs):
        self.curves: tp.OrderedDict[str, Curve] = OrderedDict()
        self.params = []
        for param_name in start:
            _, param_type = param_name.rsplit(".", 1)
            require_grad = param_type not in self.frozen_params
            self.curves[param_name] = curve_type(start[param_name],
                                                 end[param_name],
                                                 require_grad,
                                                 **curve_kwargs)
            self.params.extend(self.curves[param_name].parameters())

    def parameters(self):
        return self.params

    def get_point(self, t):
        return OrderedDict([(param_name, curve.get_point(t))
                            for param_name, curve in self.curves.items()])
