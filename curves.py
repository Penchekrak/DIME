import string

import numpy as np
import torch
import torch.nn as nn

from math import trunc

from torch.nn.parameter import Parameter


def point_on_line(start, end, t):
    return (1 - t) * start + t * end


class Curve(nn.Module):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor) -> None:
        super(Curve, self).__init__()
        self.start = start.data
        self.end = end.data

    def get_inner_point(self, t: float) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t_batch: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.get_inner_point(t) for t in t_batch])


class Polyline2(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 point: torch.Tensor = None) -> None:
        super(Polyline2, self).__init__(start, end)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point)

    def get_inner_point(self, t: float) -> torch.Tensor:
        if t < .5:
            return point_on_line(self.start, self.point, 2 * t)
        return point_on_line(self.point, self.end, 2 * t - 1)


class PolylineN(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 n_nodes: int) -> None:
        super(PolylineN, self).__init__(start, end)
        self.n_nodes = n_nodes

        points = []
        for i in range(1, n_nodes):
            points.append(Parameter(point_on_line(start, end, i / n_nodes)))
        self.params = nn.ParameterList(points)
        self._segments = [self.start] + points + [self.end]

    def get_inner_point(self, t: torch.Tensor) -> torch.Tensor:
        start_ix = torch.trunc(self.n_nodes * t).int()
        end_ix = torch.ceil(self.n_nodes * t).int()
        return point_on_line(self._segments[start_ix],
                             self._segments[end_ix],
                             t * self.n_nodes - start_ix)


class QuadraticBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 point: torch.Tensor = None) -> None:
        super(QuadraticBezier, self).__init__(start, end)
        point = point or point_on_line(start, end, .5)
        self.point = Parameter(point)

    def get_inner_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 2 * self.start + \
               2 * t * (1 - t) * self.point + \
               (1 - t) ** 2 * self.end


class CubicBezier(Curve):
    def __init__(self,
                 start: torch.Tensor,
                 end: torch.Tensor,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None) -> None:
        super(CubicBezier, self).__init__(start, end)
        p1 = p1 or point_on_line(start, end, .33)
        p2 = p2 or point_on_line(start, end, .67)
        self.p1 = Parameter(p1)
        self.p2 = Parameter(p2)

    def get_inner_point(self, t: torch.Tensor) -> torch.Tensor:
        return t ** 3 * self.start + \
               3 * t * (1 - t) ** 2 * self.p1 + \
               3 * t ** 2 * (1 - t) * self.p2 + \
               (1 - t) ** 3 * self.end
