from .. import _ops as NNOps
from olaf.autograd import Tensor, apply_op, parse_key
from olaf.dtypes import is_int


def embedding(x: Tensor, emb_table: Tensor) -> Tensor:
    assert is_int(x.dtype)
    key = parse_key(x)
    return apply_op(NNOps.Embedding, emb_table, key=key)
