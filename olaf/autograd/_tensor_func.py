from typing import Any, Optional

from ._tensor import *
from ..dtypes import *
from .._backends import *
from ._ops import movement_ops as MOps

__all__ = [
    "tensor",
    "arange",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "full",
    "full_like",
    "randi",
    "randi_like",
    "randn",
    "randn_like",
    "randu",
    "randu_like",
    "randperm",
    "concat",
    "stack",
    "where"
]


def _parse_factory_kwargs(
        device: Optional[DeviceLike], dtype: Optional[DType]
) -> tuple[Device, DType]:
    device = select_device(device)
    dtype = select_dtype(dtype)
    return device, dtype


def tensor(
        data: Any,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device, _ = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.asarray(data, dtype)
    assert not req_grad or is_float(data.dtype), "Tensors that req. grad must be float."
    return Tensor(data, req_grad=req_grad)


def arange(
        stop: float,
        start: float = 0,
        step: float = 1,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = int64,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.arange(start, stop, step, dtype)
    return Tensor(data, req_grad=req_grad)


def ones(
        *shape: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.ones(shape, dtype)
    return Tensor(data, req_grad=req_grad)


def ones_like(x: Tensor, req_grad: bool = False) -> Tensor:
    return ones(*x.shape, device=x.device, dtype=x.dtype, req_grad=req_grad)


def zeros(
        *shape: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.zeros(shape, dtype)
    return Tensor(data, req_grad=req_grad)


def zeros_like(x: Tensor, req_grad: bool = False) -> Tensor:
    return zeros(*x.shape, device=x.device, dtype=x.dtype, req_grad=req_grad)


def full(
        *shape: int,
        value: Scalar,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.full(shape, value, dtype)
    return Tensor(data, req_grad=req_grad)


def full_like(x: Tensor, value: Scalar, req_grad: bool = False) -> Tensor:
    return full(
        *x.shape, value=value, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def randi(
        *shape: int,
        low: int,
        high: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = int64,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.randint(low, high, shape, dtype)
    return Tensor(data, req_grad=req_grad)


def randi_like(x: Tensor, low: int, high: int, req_grad: bool = False) -> Tensor:
    return randi(
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
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.normal(mean, var, shape).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randn_like(
        x: Tensor, mean: float = 0, var: float = 1, req_grad: bool = False
) -> Tensor:
    return randn(
        *x.shape, mean=mean, var=var, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def randu(
        *shape: int,
        low: float = -1,
        high: float = 1,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = None,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.uniform(low, high, shape).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def randu_like(
        x: Tensor, low: float = -1, high: float = 1, req_grad: bool = False
) -> Tensor:
    return randu(
        *x.shape, low=low, high=high, device=x.device, dtype=x.dtype, req_grad=req_grad
    )


def randperm(
        n: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[DType] = int64,
        req_grad: bool = False,
) -> Tensor:
    device, dtype = _parse_factory_kwargs(device, dtype)
    with device:
        data = device.xp.random.permutation(n).astype(dtype)
    return Tensor(data, req_grad=req_grad)


def concat(*tensors: Tensor, dim: int = 0):
    return apply_op(MOps.Concat, *tensors, dim=dim)


def stack(*tensors: Tensor, dim: int = 0):
    return apply_op(MOps.Stack, *tensors, dim=dim)


def where(condition: Tensor, x1: Tensor | Scalar, x2: Tensor | Scalar) -> Tensor:
    x1 = x1 if isinstance(x1, Tensor) else tensor(x1, condition.device, dtype=float32)
    x2 = x2 if isinstance(x2, Tensor) else tensor(x2, condition.device, dtype=float32)
    return apply_op(MOps.Where, condition, x1, x2)
