from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op


def batchnorm(
        x: Tensor,
        w: Tensor,
        b: Tensor,
        rmean: Tensor,
        rvar: Tensor,
        momentum: float = 0.1,
        eps: float = 1e-5,
        training: bool = False,
) -> Tensor:
    return apply_op(
        NNOps.BatchNorm,
        x,
        rmean,
        rvar,
        w,
        b,
        momentum=momentum,
        eps=eps,
        training=training,
    )


def layernorm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    return apply_op(NNOps.LayerNorm, x, w, b, eps=eps)
