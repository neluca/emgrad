from typing import Optional
from ..autograd import Tensor

__all__ = ["Parameter", "Buffer"]


class Parameter(Tensor):
    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data, req_grad=True, label=label)


class Buffer(Tensor):
    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data, label=label)
