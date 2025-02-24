from typing import Any, Literal, Optional
from olaf.autograd.ops import Op
from olaf.dtypes import ArrayLike, DType, Shape
from olaf._backends import (
    Device,
    get_array_device
)

__all__ = ["Tensor"]


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
