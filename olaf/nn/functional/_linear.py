from typing import Optional
from .. import _ops as NNOps
from olaf.autograd import Tensor, apply_op


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    return apply_op(NNOps.Linear, x, w, b)
