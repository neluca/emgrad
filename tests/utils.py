import numpy as np
import torch
from torch import Tensor as _Tensor

import emgrad as ed
from emgrad import *

np.random.seed(0)


def close(ol_in: ArrayLike, torch_in: _Tensor, tol: float = 1e-5):
    return np.allclose(ol_in, torch_in.detach().numpy(), atol=tol, rtol=tol)


def get_random_floats(shape: ShapeLike, req_grad: bool = True):
    x = ed.random.randn(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_random_positive_floats(shape: ShapeLike, req_grad: bool = True):
    x = ed.random.randn(*shape).abs()
    x.req_grad = req_grad
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_random_ints(shape: ShapeLike, low: int, high: int):
    x = ed.random.randint(*shape, low=low, high=high, dtype=int64)
    torch_x = torch.tensor(x.data)
    return x, torch_x


def get_random_bools(shape: ShapeLike):
    x = ed.random.randn(*shape) < 0
    torch_x = torch.tensor(x.data)
    return x, torch_x


def get_ones(shape: ShapeLike, req_grad: bool = False):
    x = ed.autograd.ones(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x


def get_zeros(shape: ShapeLike, req_grad: bool = False):
    x = ed.autograd.zeros(*shape, req_grad=req_grad)
    torch_x = torch.tensor(x.data, requires_grad=req_grad)
    return x, torch_x
