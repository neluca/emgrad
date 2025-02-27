from .. import _ops as NNOps
from emgrad.autograd import Tensor, apply_op, parse_key
from emgrad.dtypes import is_integer


def embedding(x: Tensor, emb_table: Tensor) -> Tensor:
    assert is_integer(x.dtype)
    key = parse_key(x)
    return apply_op(NNOps.Embedding, emb_table, key=key)
