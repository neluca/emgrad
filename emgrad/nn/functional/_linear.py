from typing import Optional
from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    return apply_op(NNOps.Linear, x, w, b)
