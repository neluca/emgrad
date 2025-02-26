from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op


def maxpool2d(x: Tensor, window_size: int = 2) -> Tensor:
    return apply_op(NNOps.MaxPool2D, x, window_size=window_size)
