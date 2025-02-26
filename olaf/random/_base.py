from typing import Any, Optional

from ..autograd import *
from ..dtypes import *
from .._backends import *

__all__ = [
    "seed",
    "rand",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "uniform",
    "uniform_like",
    "permutation",
]


def seed(num: int) -> None:
    set_random_seed(num)


def rand(
        *shape: int, device: Optional[DeviceLike] = None, dtype: Optional[DType] = None, req_grad: bool = False,
) -> Tensor:
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.xp.random.rand(shape).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randint(
        *shape: int,
        low: int,
        high: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = int64,
        req_grad: bool = False,
) -> Tensor:
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.xp.random.randint(low, high, shape, dtype)
    return Tensor(data, req_grad=req_grad)


def randint_like(x: Tensor, low: int, high: int, req_grad: bool = False) -> Tensor:
    return randint(
        *x.shape, low=low, high=high, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def randn(
        *shape: int,
        mean: float = 0,
        var: float = 1,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.xp.random.normal(mean, var, shape).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randn_like(
        x: Tensor, mean: float = 0, var: float = 1, req_grad: bool = False
) -> Tensor:
    return randn(
        *x.shape, mean=mean, var=var, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def uniform(
        *shape: int,
        low: float = -1,
        high: float = 1,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.xp.random.uniform(low, high, shape).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def uniform_like(
        x: Tensor, low: float = -1, high: float = 1, req_grad: bool = False
) -> Tensor:
    return uniform(
        *x.shape, low=low, high=high, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def permutation(
        n: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = int64,
        req_grad: bool = False,
) -> Tensor:
    device = select_device(device)
    dtype = select_dtype(dtype)
    with device:
        data = device.xp.random.permutation(n).astype(dtype)
    return Tensor(data, req_grad=req_grad)
