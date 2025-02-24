from typing import Any, Literal, Optional
from olaf.autograd.ops import Op
from olaf.autograd.ops import unary_ops as UOps
from olaf.dtypes import ArrayLike, DType, Shape
from olaf._backends import (
    Device,
    get_array_device
)

__all__ = ["Tensor"]

_autograd_tracking_active = True


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
        """Returns the tensor label."""
        if self._label:
            return self._label
        if self.ctx is not None:
            return self.ctx.name
        return self.__class__.__name__

    @property
    def device(self) -> Device:
        """Returns the device on which the tensor data is stored."""
        return get_array_device(self.data)

    @property
    def dtype(self) -> DType:
        """Returns the data type of the tensor."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        """Returns the shape of the tensor."""
        return Shape(self.data.shape)

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self.data.size

    def abs(self) -> "Tensor":
        """Computes the element-wise absolute value of the tensor.

        Returns:
            Tensor: A tensor with absolute values of the elements.
        """
        return apply_op(UOps.Abs, self)

    def exp(self) -> "Tensor":
        """Computes the element-wise exponential function.

        Returns:
            Tensor: A tensor where each element is `e` raised to the power of the
                corresponding element.
        """
        return apply_op(UOps.Exp, self)


def apply_op(op: type[Op], *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    tensor_args = [t for t in tensors if t is not None]
    device = tensor_args[0].device
    ctx = op(device, kwargs)

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
