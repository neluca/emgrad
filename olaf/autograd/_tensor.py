from __future__ import annotations

from typing import Any, Optional, Iterator
import numpy as np
from olaf.operators import Op
from ._ops import unary_ops as UOps
from ._ops import binary_ops as BOps
from ._ops import reduce_ops as ROps
from ._ops import movement_ops as MOps
from ..dtypes import *
from .._backends import *

__all__ = ["Tensor", "apply_op", "no_grad", "parse_key"]

_autograd_tracking_active: bool = True


class no_grad:
    def __enter__(self):
        global _autograd_tracking_active
        _autograd_tracking_active = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_tracking_active
        _autograd_tracking_active = True


def parse_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Tensor) else k for k in key)
    if isinstance(key, Tensor):
        return key.data
    return key


def _get_shape_diff(shape1: ShapeLike, shape2: ShapeLike) -> Dim:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


class Tensor:
    def __init__(
            self,
            data: ArrayLike,
            ctx: Optional[Op] = None,
            src: Optional[tuple[Optional["Tensor"], ...]] = None,
            req_grad: bool = False,
            label: Optional[str] = None,
    ) -> None:
        self.data = data
        self.ctx = ctx
        self.src = src
        self.req_grad = req_grad
        self._label = label
        self.grad: Optional[ArrayLike] = None

    @property
    def label(self) -> str:
        if self._label:
            return self._label
        if self.ctx is not None:
            return self.ctx.name
        return self.__class__.__name__

    @property
    def device(self) -> Device:
        return get_array_device(self.data)

    @property
    def dtype(self) -> DType:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        return Shape(self.data.shape)

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose(-2, -1)

    def __add__(self, x: Scalar | "Tensor") -> "Tensor":
        return self.add(x)

    __radd__ = __add__

    def __sub__(self, x: Scalar | "Tensor") -> "Tensor":
        return self.sub(x)

    def __rsub__(self, x: Scalar) -> "Tensor":
        return self.align(x).sub(self)

    def __mul__(self, x: Scalar | "Tensor") -> "Tensor":
        return self.mul(x)

    __rmul__ = __mul__

    def __truediv__(self, x: Scalar | "Tensor") -> "Tensor":
        return self.truediv(x)

    def __rtruediv__(self, x: Scalar) -> "Tensor":
        return self.align(x).truediv(self)

    def __matmul__(self, x: "Tensor") -> "Tensor":
        return self.dot(x)

    def __pow__(self, x: Scalar) -> "Tensor":
        return self.pow(x)

    def __neg__(self) -> "Tensor":
        return self.mul(-1)

    def __eq__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data == self.align(x).data)

    def __ne__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data != self.align(x).data)

    def __lt__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data < self.align(x).data)

    def __gt__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data > self.align(x).data)

    def __le__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data <= self.align(x).data)

    def __ge__(self, x: Scalar | "Tensor") -> "Tensor":
        return Tensor(self.data >= self.align(x).data)

    def __getitem__(self, key: Any) -> "Tensor":
        return self.select(key)

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", dtype={self.dtype}, device={self.device}"
        suffix += f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return prefix + array_to_string(self.data, prefix) + suffix

    def __len__(self) -> int:
        return 0 if len(self.shape) == 0 else self.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def accumulate_grad(self, dy: ArrayLike) -> None:
        assert dy.dtype == float32, f"Gradient has invalid dtype {dy.dtype}"
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[ArrayLike] = None):
        assert self.req_grad, "Node is not part of a autograd graph."
        assert self.grad is None, "Cannot run backward multiple times."

        # set node grad
        if dy is None:
            self.grad = self.device.xp.ones(self.shape, dtype=float32)
        else:
            assert isinstance(dy, ArrayLike), "Gradient must be an array."
            self.grad = dy

        # run backward through traced graph
        node_queue = _get_node_tree_dfs(self, [], set())
        for node in reversed(node_queue):
            assert node.ctx is not None, "Node has no function context."
            assert node.src is not None, "Node has no source nodes."
            assert node.grad is not None, "Node has no grad."
            grads = node.ctx.backward(node.grad)
            for src_tensor, grad in zip(node.src, grads):
                if src_tensor is None or not src_tensor.req_grad:
                    continue
                grad = _undo_broadcast(grad, src_tensor.shape)
                src_tensor.accumulate_grad(grad)

            # clear context of intermediate nodes
            node.grad, node.ctx, node.src = None, None, None

    def abs(self) -> "Tensor":
        return apply_op(UOps.Abs, self)

    def exp(self) -> "Tensor":
        return apply_op(UOps.Exp, self)

    def log(self) -> "Tensor":
        return apply_op(UOps.Log, self)

    def pow(self, exponent: Scalar) -> "Tensor":
        return apply_op(UOps.Pow, self, exp=exponent)

    def sqrt(self) -> "Tensor":
        return apply_op(UOps.Sqrt, self)

    def tanh(self) -> "Tensor":
        return apply_op(UOps.Tanh, self)

    def tril(self, diag: int = 0) -> "Tensor":
        return apply_op(UOps.Tril, self, diag=diag)

    def triu(self, diag: int = 0) -> "Tensor":
        return apply_op(UOps.Triu, self, diag=diag)

    def add(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Add, self, self.align(x))

    def sub(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Sub, self, self.align(x))

    def mul(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Mul, self, self.align(x))

    def truediv(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Div, self, self.align(x))

    def dot(self, x: "Tensor") -> "Tensor":
        return apply_op(BOps.Dot, self, x)

    def maximum(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Maximum, self, self.align(x))

    def minimum(self, x: Scalar | "Tensor") -> "Tensor":
        return apply_op(BOps.Minimum, self, self.align(x))

    def sum(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> "Tensor":
        return apply_op(ROps.Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> "Tensor":
        return apply_op(ROps.Mean, self, dim=dim, keepdims=keepdims)

    def var(
            self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> "Tensor":
        return apply_op(ROps.Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(
            self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> "Tensor":
        return apply_op(ROps.Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def max(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> "Tensor":
        return apply_op(ROps.Max, self, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> "Tensor":
        return apply_op(ROps.Min, self, dim=dim, keepdims=keepdims)

    def reshape(self, *shape: int) -> "Tensor":
        return apply_op(MOps.Reshape, self, shape=shape)

    def expand(self, *shape: int) -> "Tensor":
        return apply_op(MOps.Expand, self, shape=shape)

    def select(self, key: Any) -> "Tensor":
        key = parse_key(key)
        return apply_op(MOps.Select, self, key=key)

    def _split(self, key: Any) -> "Tensor":
        key = parse_key(key)
        return apply_op(MOps.Split, self, key=key)

    def split(self, split_size: int, *, dim: int = -1) -> list["Tensor"]:
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        return [
            self._split(pre_dim_slice + (slice(i, i + split_size),) + post_dim_slice)
            for i in range(0, self.shape[dim], split_size)
        ]

    def squeeze(self) -> "Tensor":
        non_singular_dims = tuple(d for d in self.shape if d > 1)
        if len(non_singular_dims) == self.ndim:
            return self
        return apply_op(MOps.Squeeze, self, shape=non_singular_dims)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> "Tensor":
        return apply_op(MOps.Transpose, self, dim1=dim1, dim2=dim2)

    def view(self, *shape: int) -> "Tensor":
        if shape == self.shape:
            return self
        return apply_op(MOps.View, self, shape=shape)

    def as_type(self, dtype: DType) -> "Tensor":
        if self.dtype == dtype:
            return self
        data: ArrayLike = self.data.astype(dtype)
        if self.req_grad:
            assert is_float(dtype), "Cannot change autograd node dtype to non float."
            arr = Tensor(data, self.ctx, self.src, self.req_grad)
            if self.grad is not None:
                arr.grad = self.grad.astype(dtype)
            return arr
        return Tensor(data)

    def int(self) -> "Tensor":
        return self.as_type(int32)

    def long(self) -> "Tensor":
        return self.as_type(int64)

    def float(self) -> "Tensor":
        return self.as_type(float32)

    def to(self, device: DeviceLike) -> "Tensor":
        device = parse_device(device)
        if device == self.device:
            return self
        data = move_to_device(self.data, device)
        if self.req_grad:
            arr = Tensor(data, self.ctx, self.src, self.req_grad)
            if self.grad is not None:
                arr.grad = move_to_device(self.grad, device)
            return arr
        return Tensor(data)

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def cuda(self) -> "Tensor":
        return self.to("cuda")

    def ito(self, device: DeviceLike) -> None:
        device = parse_device(device)
        if self.device == device:
            return
        self.data = move_to_device(self.data, device)
        if self.grad is not None:
            self.grad = move_to_device(self.grad, device)

    def item(self) -> Any:
        return self.data.item()

    def contiguous(self) -> "Tensor":
        data = self.device.xp.ascontiguousarray(self.data)
        return Tensor(data, self.ctx, self.src, self.req_grad)

    def align(self, x: Scalar | "Tensor") -> "Tensor":
        if isinstance(x, Tensor):
            return x.as_type(self.dtype)
        return Tensor(self.device.xp.asarray(x, dtype=self.dtype))

    def numpy(self) -> np.ndarray:
        return self.cpu().data

    def iter_dim(self, dim: int) -> Iterator["Tensor"]:
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        for i in range(self.shape[dim]):
            yield self[pre_dim_slice + (i,) + post_dim_slice]

    def argmax(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> "Tensor":
        return Tensor(self.device.xp.argmax(self.data, axis=dim, keepdims=keepdims))


def apply_op(op: type[Op], *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    tensor_args = [t for t in tensors if t is not None]
    device = tensor_args[0].device
    ctx = op(device)

    # compute forward pass
    fwd_args = [t.data if t is not None else None for t in tensors]
    with device:
        data = ctx.forward(*fwd_args, **kwargs)

    # return result node with autograd context
    result_req_grad = any(t.req_grad for t in tensor_args)
    if _autograd_tracking_active and result_req_grad:
        return Tensor(data, ctx=ctx, src=tensors, req_grad=True)

    # return result node without autograd context
    return Tensor(data)


def _undo_broadcast(grad: ArrayLike, target_shape: ShapeLike) -> ArrayLike:
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        shape = _get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(shape, keepdims=True)
    else:
        data_shape = Shape((1,) * (grad.ndim - target_ndim) + target_shape)
        shape = _get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(shape)

    return grad.reshape(target_shape)


def _get_node_tree_dfs(node: Tensor, queue: list[Tensor], visited: set) -> list[Tensor]:
    if node not in visited:
        visited.add(node)
        if not node.src:
            return []
        for p in node.src:
            if p is not None and p.req_grad:
                _ = _get_node_tree_dfs(p, queue, visited)
        queue.append(node)
    return queue
