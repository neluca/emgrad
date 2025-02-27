from typing import Any, Optional

from ._backends import set_nn_device, get_nn_device
from .autograd import Tensor
from . import autograd
from .dtypes import DType, Scalar


def set_device(device: str) -> None:
    set_nn_device(device)


def tensor(data: Any, dtype: Optional[DType] = None, req_grad: bool = True) -> Tensor:
    return autograd.tensor(data, get_nn_device(), dtype, req_grad)


def arange(
        stop: float,
        start: float = 0,
        step: float = 1,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    return autograd.arange(stop, start, step, get_nn_device(), dtype, req_grad)


def ones(
        *shape: int,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    return autograd.ones(*shape, device=get_nn_device(), dtype=dtype, req_grad=req_grad)


def ones_like(x: Tensor, req_grad: bool = False) -> Tensor:
    return ones(*x.shape, dtype=x.dtype, req_grad=req_grad)


def zeros(
        *shape: int,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    return autograd.zeros(*shape, device=get_nn_device(), dtype=dtype, req_grad=req_grad)


def zeros_like(x: Tensor, req_grad: bool = False) -> Tensor:
    return zeros(*x.shape, dtype=x.dtype, req_grad=req_grad)


def full(
        *shape: int,
        value: Scalar = 1,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    return autograd.full(*shape,
                         value=value,
                         device=get_nn_device(),
                         dtype=dtype,
                         req_grad=req_grad)


def full_like(x: Tensor, value: Scalar, req_grad: bool = False) -> Tensor:
    return full(*x.shape, value=value, dtype=x.dtype, req_grad=req_grad)
