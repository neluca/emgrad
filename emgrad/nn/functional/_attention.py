from typing import Optional
from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op


def scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0,
) -> Tensor:
    return apply_op(NNOps.ScaledDotProductAttention, q, k, v, mask, dropout_p=dropout_p)
