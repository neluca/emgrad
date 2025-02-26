from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return x
    return apply_op(NNOps.Dropout, x, p=p)
